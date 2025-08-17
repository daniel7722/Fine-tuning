import random
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
from util.sim_load_data import extract_vision_dataset, extract_audio_dataset

@tf.keras.utils.register_keras_serializable()
class ComputeMaskLayer(layers.Layer):
    def call(self, x):
        # Produces mask of shape (B, T, 1) where 1 marks non-zero embedding rows
        return tf.expand_dims(
            tf.cast(tf.reduce_any(tf.not_equal(x, 0.0), axis=-1), tf.float32),
            axis=-1,
        )

    def compute_output_shape(self, input_shape):
        # (B, T, D) -> (B, T, 1)
        return (input_shape[0], input_shape[1], 1)

    def get_config(self):
        return {}


@tf.keras.utils.register_keras_serializable()
class NegInfFromMask(layers.Layer):
    def call(self, m):
        # Input mask m is (B, T, 1) in {0,1}. Return (1-m)*1e9 with same shape.
        return (1.0 - m) * 1e9

    def compute_output_shape(self, input_shape):
        # Shape preserving: (B, T, 1)
        return input_shape

    def get_config(self):
        return {}


@tf.keras.utils.register_keras_serializable()
class TemporalSum1D(layers.Layer):
    def call(self, x):
        # Reduce over time axis 1: (B, T, D) -> (B, D)
        return tf.reduce_sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        # (B, T, D) -> (B, D)
        return (input_shape[0], input_shape[2])

    def get_config(self):
        return {}

TIMESTEPS = 100
FEATURE_DIM = 64

class Agent(ABC): 

	def __init__(self, agent_id, class_count=2, ema_alpha=0.3):
		self.agent_id = agent_id
		self.class_count = class_count
		self.hedge_weight = tf.Variable(float(1.0/class_count), trainable=False)
		self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)  
		self.loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


	@abstractmethod
	def emit(self, episode):
		"""
		Single round: emit softmax + current trust.
		"""

	@abstractmethod
	def train_step(self, event, label): 
		"""
		Train step for the agent.
		x: input data
		y: ground truth label
		"""

class VisionAgent(Agent): 
	def __init__(self, agent_id, class_count=28): 
		super().__init__(agent_id, class_count)
		self.modality_id = 0
		self.backbone = tf.keras.applications.EfficientNetB0(
			weights='imagenet',
			include_top=False,
			pooling='avg',
		)
		self.classifier = tf.keras.Sequential([
			layers.Dense(64, activation='relu'),
			layers.Dense(class_count)
		])
		self.model = tf.keras.Sequential([self.backbone, self.classifier])

	def emit(self, data): 
		image = data["vision_data"] # Expected to be preprocessed image tensor,
		logits = self.model(tf.expand_dims(image, axis=0), training=False) # Add batch dimension
		belief = tf.nn.softmax(logits).numpy()[0]
		assert belief.shape[0] == self.class_count, f"Belief output shape mismatch: got {belief.shape}"
		return {
			"agent_id": self.agent_id,
			"belief": belief.tolist(),
			"prediction": int(np.argmax(belief)),
			"correct": (int(np.argmax(belief)) == data["label"]),
			"modality_id": self.modality_id,
			"hedge_weight": self.hedge_weight.numpy(), 
		}

	def pretrain(self, train_data, epochs=10, batch_size=32):
		"""
		Pre-train the vision agent on the training data.
		"""	
		self.backbone.trainable = False
		dataset = extract_vision_dataset(train_data).shuffle(10000).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
		self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=["accuracy"])
		callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
		self.model.fit(dataset, epochs=epochs, verbose=1, callbacks=[callback])

	def validate_pretraining(self, val_data, batch_size=32):
		"""
		Validate the pre-trained vision agent on the validation data.
		"""
		dataset = extract_vision_dataset(val_data).batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
		results = self.model.evaluate(dataset, verbose=1)
		return results[0]  # return loss
	
	def train_step(self, event, label):
		"""
		Train step for the vision agent.
		x: input image tensor
		y: ground truth label
		"""
		x = event["vision_data"]
		with tf.GradientTape() as tape:
			logits = self.model(tf.expand_dims(x, axis=0))
			loss = self.loss_fn(tf.convert_to_tensor([label]), logits)
		grads = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
		return float(loss.numpy())


class AudioAgent(Agent): 
	def __init__(self, agent_id, class_count=28):
		super().__init__(agent_id, class_count)
		self.modality_id = 1
		self.vggish = hub.KerasLayer(
			"https://tfhub.dev/google/vggish/1",
			input_shape=[],
			dtype=tf.float32,
			trainable=False
		)

		# Build audio classifier with Functional API (BiLSTM + attention pooling, proper Keras masking)
		reg = tf.keras.regularizers.l2(1e-4)
		inp = tf.keras.Input(shape=(None, 128))  # (B, T, 128)

		# Masking for padded embeddings
		masked = layers.Masking(mask_value=0.0)(inp)  # (B, T, 128)

		# Sequence encoder with L2 regularization
		h = layers.Bidirectional(
			layers.LSTM(128, return_sequences=True, kernel_regularizer=reg)
		)(masked)  # (B, T, 256)

		# Attention mechanism using Keras layers
		attn_logits = layers.Dense(1, kernel_regularizer=reg)(h)  # (B, T, 1)

		mask = ComputeMaskLayer(name="mask")(inp)

		# Large negative bias for padded positions (via Keras layers)
		neg_inf = NegInfFromMask(name="neg_inf")(mask)
		attn_logits_masked = layers.Subtract(name="mask_attn_logits")([attn_logits, neg_inf])

		attn = layers.Softmax(axis=1, name="attn_weights")(attn_logits_masked)  # (B, T, 1)

		# Context vector: sum_t (h_t * a_t)
		weighted = layers.Multiply()([h, attn])  # (B, T, 256)
		context = TemporalSum1D(name="attn_pool")(weighted)  # (B, 256)

		x = layers.Dropout(0.5)(context)
		out = layers.Dense(class_count, kernel_regularizer=reg)(x)  # (B, C)

		self.model = tf.keras.Model(inp, out, name="audio_bilstm_attn")

	@tf.function
	def _rand_gain(self, x):
		db = tf.random.uniform([], minval=-3.0, maxval=3.0)
		gain = tf.pow(10.0, db / 20.0)
		return x * gain

	@tf.function
	def _time_shift(self, x):
		sr = 16000
		max_shift = tf.cast(0.2 * tf.cast(sr, tf.float32), tf.int32)
		shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
		return tf.roll(x, shift=shift, axis=0)

	@tf.function
	def _add_noise(self, x):
		std = tf.math.reduce_std(x)
		noise = tf.random.normal(tf.shape(x), stddev=tf.maximum(1e-4, 0.03 * std))
		return x + noise

	@tf.function
	def _maybe_aug(self, x):
		def apply_or_not(fn, sig):
			return tf.cond(tf.random.uniform([]) < 0.5, lambda: fn(sig), lambda: sig)
		x = apply_or_not(self._rand_gain, x)
		x = apply_or_not(self._time_shift, x)
		x = apply_or_not(self._add_noise, x)
		return tf.clip_by_value(x, -1.0, 1.0)

	@tf.function
	def preprocess_and_embed(self, waveform, augment=False):
		# Cast & ensure 1-D
		x = tf.cast(waveform, tf.float32)
		x = tf.reshape(x, [-1])
		x = tf.clip_by_value(x, -1.0, 1.0)
		x = tf.cond(tf.convert_to_tensor(augment), lambda: self._maybe_aug(x), lambda: x)
		emb = self.vggish(x)
		emb.set_shape([None, 128])
		return emb

	def extract_embedding(self, audio_waveform):
		return self.preprocess_and_embed(audio_waveform, augment=False)

	def emit(self, data):
		"""Use the exact dataset → preprocess → padded_batch → model path for a single example."""
		# 1) Wrap the incoming dict into a 1-element Dataset
		single_ds = tf.data.Dataset.from_tensors({
			"audio_waveform": data["audio_waveform"],
			"label": data["label"],
		})

		# 2) Reuse the exact loader + preprocessing; NO augmentation at inference
		ds = (
			extract_audio_dataset(single_ds)
			.map(lambda x, y: (self.preprocess_and_embed(x, augment=False), y),
				num_parallel_calls=tf.data.AUTOTUNE)
			.padded_batch(1, padded_shapes=([None, 128], []), drop_remainder=True)
			.prefetch(tf.data.AUTOTUNE)
		)

		# 3) Materialize the single batch and run the model
		(emb_batch, y_batch) = next(iter(ds))                 # emb_batch: (1, T, 128), y_batch: (1,)
		logits = self.model(emb_batch, training=False)        # (1, C)

		belief = tf.nn.softmax(logits)[0].numpy()
		pred   = int(np.argmax(belief))
		gt     = int(y_batch[0].numpy())

		return {
			"agent_id": self.agent_id,
			"belief": belief.tolist(),
			"prediction": pred,
			"correct": (pred == gt),
			"modality_id": self.modality_id,
			"hedge_weight": self.hedge_weight.numpy(),
		}

	def pretrain(self, train_data, val_data=None, epochs=10, batch_size=32):
		"""Pre-train from raw waveforms with validation monitoring and light augmentation."""
		sr = 16000

		train_ds = (
			extract_audio_dataset(train_data)
			.map(lambda x, y: (self.preprocess_and_embed(x, augment=False), y), num_parallel_calls=tf.data.AUTOTUNE)
			.cache()
			.shuffle(10000)
			.padded_batch(batch_size, padded_shapes=([None, 128], []), drop_remainder=True)
			.prefetch(tf.data.AUTOTUNE)
		)

		# --- validation dataset (no augmentation, no shuffle) ---
		val_ds = None
		if val_data is not None:
			val_ds = (
				extract_audio_dataset(val_data)
				.map(lambda x, y: (self.preprocess_and_embed(x, augment=False), y), num_parallel_calls=tf.data.AUTOTUNE)
				.cache()
				.padded_batch(batch_size, padded_shapes=([None, 128], []), drop_remainder=True)
				.prefetch(tf.data.AUTOTUNE)
			)

		# --- class weights from label distribution for balancing ---
		def compute_class_weights(ds):
			counts = np.zeros(self.class_count, dtype=np.int64)
			it = (
				extract_audio_dataset(ds)
				.map(lambda x, y: tf.cast(y, tf.int32))
				.batch(4096)
				.as_numpy_iterator()
			)
			for y in it:
				vals, cts = np.unique(y, return_counts=True)
				for v, c in zip(vals, cts):
					if 0 <= v < self.class_count:
						counts[int(v)] += int(c)
			total = int(counts.sum()) or 1
			return {i: float(total / (self.class_count * max(1, counts[i]))) for i in range(self.class_count)}

		class_weight = compute_class_weights(train_data)

		# compile with from_logits=True, no label_smoothing
		self.model.compile(
			optimizer=self.optimizer,
			loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			metrics=["accuracy"]
		)

		callbacks = [
			tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss' if val_ds is not None else 'loss',
			                                     factor=0.5, patience=2, verbose=1),
			tf.keras.callbacks.EarlyStopping(monitor='val_loss' if val_ds is not None else 'loss',
			                                 patience=4, restore_best_weights=True)
		]

		self.model.fit(
			train_ds,
			epochs=epochs,
			verbose=1,
			callbacks=callbacks,
			class_weight=class_weight,
			validation_data=val_ds,
		)

	def validate_pretraining(self, val_data, batch_size=32):
		"""
		Validate the pre-trained audio agent on the validation data.
		"""

		dataset = (
			extract_audio_dataset(val_data)
			.map(lambda x, y: (self.preprocess_and_embed(x, augment=False), y), num_parallel_calls=tf.data.AUTOTUNE)
			.cache()
			.padded_batch(batch_size, padded_shapes=([None, 128], []), drop_remainder=True)
			.prefetch(tf.data.AUTOTUNE)
		)

		# ensure compiled for evaluation
		if not hasattr(self.model, 'loss') or self.model.loss is None:
			self.model.compile(optimizer=self.optimizer,
			                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
			                   metrics=["accuracy"]) 
		results = self.model.evaluate(dataset, verbose=1)
		return results[0]

	def train_step(self, event, label):
		"""
		Train step for the audio agent.
		x: input audio tensor
		y: ground truth label
		"""
		waveform = event["audio_waveform"]
		with tf.GradientTape() as tape:
			emb = self.preprocess_and_embed(waveform, augment=False)
			emb = tf.expand_dims(emb, axis=0)  # add batch dimension
			logits = self.model(emb)
			loss = self.loss_fn(tf.convert_to_tensor([label]), logits)
		grads = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
		return float(loss.numpy())
