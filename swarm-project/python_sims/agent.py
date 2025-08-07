import random
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers

TIMESTEPS = 100
FEATURE_DIM = 64

def extract_vision_dataset(data):
	return data.map(lambda x: (x["vision_data"], x["label"]), num_parallel_calls=tf.data.AUTOTUNE)

def extract_audio_dataset(data):
	return data.map(lambda x: (x["audio_data"], x["label"]), num_parallel_calls=tf.data.AUTOTUNE)

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

	def pretrain(self, train_data, epochs=10):
		"""
		Pre-train the vision agent on the training data.
		"""	
		self.backbone.trainable = False
		dataset = extract_vision_dataset(train_data).prefetch(tf.data.AUTOTUNE)
		self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=["accuracy"])
		callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
		self.model.fit(dataset, epochs=epochs, verbose=1, callbacks=[callback])

	def validate_pretraining(self, val_data):
		"""
		Validate the pre-trained vision agent on the validation data.
		"""
		dataset = extract_vision_dataset(val_data).prefetch(tf.data.AUTOTUNE)
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
		self.model = tf.keras.Sequential([
			tf.keras.Input(shape=(TIMESTEPS, FEATURE_DIM)),
			layers.Masking(mask_value=0.0),
			layers.LSTM(128, return_sequences=False, use_cudnn=False),
			layers.Dense(64, activation='relu'),
			layers.Dense(class_count)
		])

	def emit(self, event):
		audio = tf.convert_to_tensor(event["audio_data"], dtype=tf.float32)
		if len(audio.shape) == 1 or (len(audio.shape) > 1 and audio.shape[-1] != 1):
			audio = tf.expand_dims(audio, axis=-1) # Ensure shape is (H, W, 1)
		features = self.model(tf.expand_dims(audio, axis=0), training=False) # Add batch dimension
		belief = tf.nn.softmax(features).numpy()[0]
		assert belief.shape[0] == self.class_count, f"Belief output shape mismatch: got {belief.shape}"
		return {
			"agent_id": self.agent_id,
			"belief": belief.tolist(),
			"prediction": int(np.argmax(belief)),
			"correct": (int(np.argmax(belief)) == event["label"]),
			"modality_id": self.modality_id,
			"hedge_weight": self.hedge_weight.numpy(), 
		}

	def pretrain(self, train_data, epochs=10):
		"""
		Pre-train the audio agent on the training data.
		"""	
		dataset = extract_audio_dataset(train_data).prefetch(tf.data.AUTOTUNE)
		self.model.compile(optimizer=self.optimizer, loss=self.loss_fn, metrics=["accuracy"])
		callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2, restore_best_weights=True)
		self.model.fit(dataset, epochs=epochs, verbose=1, callbacks=[callback])

	def validate_pretraining(self, val_data):
		"""
		Validate the pre-trained audio agent on the validation data.
		"""
		dataset = extract_audio_dataset(val_data).prefetch(tf.data.AUTOTUNE)
		results = self.model.evaluate(dataset, verbose=1)
		return results[0]  # return loss

	def train_step(self, event, label):
		"""
		Train step for the audio agent.
		x: input audio tensor
		y: ground truth label
		"""
		x = event["audio_data"]
		with tf.GradientTape() as tape:
			logits = self.model(tf.expand_dims(x, axis=0))
			loss = self.loss_fn(tf.convert_to_tensor([label]), logits)
		grads = tape.gradient(loss, self.model.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
		return float(loss.numpy())
