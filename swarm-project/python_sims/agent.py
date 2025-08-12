import random
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_hub as hub
from util.sim_load_data import extract_vision_dataset, extract_audio_dataset

TIMESTEPS = 100
FEATURE_DIM = 64

class Agent(ABC): 

	def __init__(self, agent_id, class_count=2, ema_alpha=0.3):
		self.agent_id = agent_id
		self.class_count = class_count
		self.hedge_weight = tf.Variable(1.0, trainable=False)
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

	def emit(self, event): 
		image = event["vision_data"] # Expected to be preprocessed image tensor,
		logits = self.model(tf.expand_dims(image, axis=0), training=False) # Add batch dimension
		belief = tf.nn.softmax(logits).numpy()[0]
		assert belief.shape[0] == self.class_count, f"Belief output shape mismatch: got {belief.shape}"
		return {
			"agent_id": self.agent_id,
			"belief": belief.tolist(),
			"prediction": int(np.argmax(belief)),
			"correct": (int(np.argmax(belief)) == event["label"]),
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
	def __init__(self, agent_id, class_count=2):
		super().__init__(agent_id, class_count)
		self.modality_id = 1
		self.vggish = hub.KerasLayer(
			"https://tfhub.dev/google/vggish/1", 
			input_shape=[],
			dtype=tf.float32,
			trainable=False
		)

		# Build audio classifier with Functional API (BiLSTM + attention pooling)
		reg = tf.keras.regularizers.l2(1e-4)
		inp = tf.keras.Input(shape=(None, 128))                       # (B, T, 128)

		# Propagate mask to RNNs
		masked = layers.Masking(mask_value=0.0)(inp)                  # (B, T, 128)

		# Sequence encoder with L2 regularization
		h = layers.Bidirectional(
			layers.LSTM(128, return_sequences=True, kernel_regularizer=reg)
		)(masked)  # (B, T, 256)

		# Attention over time
		attn_logits = layers.Dense(1, kernel_regularizer=reg)(h)      # (B, T, 1)

		# Build a mask tensor with shape (B, T, 1) using Lambda so it remains symbolic
		mask = layers.Lambda(
			lambda t: tf.expand_dims(
				tf.cast(tf.reduce_any(tf.not_equal(t, 0.0), axis=-1), tf.float32),
				axis=-1,
			)
		)(inp)  # (B, T, 1)

		# Apply large negative bias to padded positions before softmax
		neg_inf = layers.Lambda(lambda m: (1.0 - m) * 1e9)(mask)
		attn_logits_masked = layers.Subtract()([attn_logits, neg_inf])

		attn = layers.Softmax(axis=1, name="attn_weights")(attn_logits_masked)  # (B, T, 1)

		# Context vector: sum_t (h_t * a_t)
		weighted = layers.Multiply()([h, attn])                       # (B, T, 256)
		context = layers.Lambda(lambda t: tf.reduce_sum(t, axis=1), name="attn_pool")(weighted)  # (B, 256)

		x = layers.Dropout(0.5)(context)
		out = layers.Dense(class_count, kernel_regularizer=reg)(x)    # (B, C)

		self.model = tf.keras.Model(inp, out, name="audio_bilstm_attn")
		# baseline: mean over time, then linear
		# class MyLayer(tf.keras.layers.Layer): 
		# 	def call(self, x): 
		# 		return tf.reduce_mean(x, axis=1)
		# inputs = tf.keras.Input(shape=(None, 128))
		# x = MyLayer()(inputs)                 # (B, 128)
		# outputs = tf.keras.layers.Dense(28)(x)
		# self.model = tf.keras.Model(inputs, outputs)
		# self.classifier = self.model

	def extract_embedding(self, audio_waveform):
		"""Takes a (possibly unknown-rank) waveform tensor and returns a [time_steps, 128] embedding."""
		audio_waveform = tf.convert_to_tensor(audio_waveform, dtype=tf.float32)
		# Ensure 1-D at runtime (handles unknown static shapes during graph tracing)
		audio_waveform = tf.reshape(audio_waveform, [-1])  # shape [num_samples]
		embedding = self.vggish(audio_waveform)
		embedding.set_shape([None, 128])  # [time_steps, 128]
		return embedding

	def emit(self, event):
		waveform = event["audio_waveform"]
		embedding = self.extract_embedding(waveform)
		embedding = tf.expand_dims(embedding, axis=0)  # batch dimension
		logits = self.model(embedding, training=False)
		belief = tf.nn.softmax(logits).numpy()[0]
		assert belief.shape[0] == self.class_count, f"Belief output shape mismatch: got {belief.shape}"
		
		return {
			"agent_id": self.agent_id,
			"belief": belief.tolist(),
			"prediction": int(np.argmax(belief)),
			"correct": (int(np.argmax(belief)) == event["label"]),
			"modality_id": self.modality_id,
			"hedge_weight": self.hedge_weight.numpy(),
		}

	def pretrain(self, train_data, val_data=None, epochs=50, batch_size=32):
		"""Pre-train from raw waveforms with validation monitoring and light augmentation."""
		sr = 16000

		# --- label smoothing loss just for training this agent ---
		# loss_with_ls = tf.keras.losses.SparseCategoricalCrossentropy(
		# 	from_logits=True, label_smoothing=0.1
		# )

		# --- augmentation functions on waveform ---
		def _rand_gain(x):
			# +/- 3 dB
			db = tf.random.uniform([], minval=-3.0, maxval=3.0)
			gain = tf.pow(10.0, db / 20.0)
			return x * gain

		def _time_shift(x):
			# +/- 0.2s
			max_shift = int(0.2 * sr)
			shift = tf.random.uniform([], -max_shift, max_shift + 1, dtype=tf.int32)
			return tf.roll(x, shift=shift, axis=0)

		def _add_noise(x):
			# low-level Gaussian noise relative to signal std
			std = tf.math.reduce_std(x)
			noise = tf.random.normal(tf.shape(x), stddev=tf.maximum(1e-4, 0.03 * std))
			return x + noise

		def _maybe_aug(x):
			# Apply each augmentation with 0.5 prob independently
			def apply_or_not(fn, sig):
				return tf.cond(tf.random.uniform([]) < 0.5, lambda: fn(sig), lambda: sig)
			x = apply_or_not(_rand_gain, x)
			x = apply_or_not(_time_shift, x)
			x = apply_or_not(_add_noise, x)
			return tf.clip_by_value(x, -1.0, 1.0)

		# pair (waveform -> embedding, label)
		def extract_and_pair(x, y, augment=False):
			if augment:
				x = _maybe_aug(x)
			emb = self.extract_embedding(x)
			emb.set_shape([None, 128])
			return emb, y

		# --- training dataset ---
		train_ds = (
			extract_audio_dataset(train_data)
			.map(lambda x, y: extract_and_pair(x, y, augment=True), num_parallel_calls=tf.data.AUTOTUNE)
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
				.map(lambda x, y: extract_and_pair(x, y, augment=False), num_parallel_calls=tf.data.AUTOTUNE)
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

		# compile with label smoothing
		self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=["accuracy"])

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
		def extract_and_pair(x, y):
			emb = self.extract_embedding(x)
			emb.set_shape([None, 128])  # ensure shape [time_steps, 128]
			return emb, y

		dataset = extract_audio_dataset(val_data) \
			.map(extract_and_pair, num_parallel_calls=tf.data.AUTOTUNE) \
			.cache() \
			.padded_batch(batch_size, padded_shapes=([None, 128], []), drop_remainder=True) \
			.prefetch(tf.data.AUTOTUNE)

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
			emb = self.extract_embedding(waveform)
			emb = tf.expand_dims(emb, axis=0)  # add batch dimension
			logits = self.model(emb)
			loss = self.loss_fn(tf.convert_to_tensor([label]), logits)
		grads = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
		return float(loss.numpy())
