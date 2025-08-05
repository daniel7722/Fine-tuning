import random
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf
from tensorflow.keras import layers

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
	def train_step(self, x, y): 
		"""
		Train step for the agent.
		x: input data
		y: ground truth label
		"""

class VisionAgent(Agent): 
	def __init__(self, agent_id, class_count=28): 
		super().__init__(agent_id, class_count)
		self.modality_id = 0
		self.model = tf.keras.applications.EfficientNetB0(
			weights='imagenet',
			include_top=False,
			pooling='avg',
		)
		self.classifier = tf.keras.Sequential([
			layers.Dense(64, activation='relu'),
			layers.Dense(class_count)
		])

	def emit(self, event): 
		image = event["vision_data"] # Expected to be preprocessed image tensor,
		features = self.model(tf.expand_dims(image, axis=0)) # Add batch dimension
		logits = self.classifier(features)
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
	
	def train_step(self, x, y):
		"""
		Train step for the vision agent.
		x: input image tensor
		y: ground truth label
		"""
		with tf.GradientTape() as tape:
			logits = self.classifier(self.model(tf.expand_dims(x, axis=0)))
			loss = self.loss_fn(tf.convert_to_tensor([y]), logits)
		grads = tape.gradient(loss, self.model.trainable_variables + self.classifier.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables + self.classifier.trainable_variables))
		return float(loss.numpy())

class AudioAgent(Agent): 
	def __init__(self, agent_id, class_count=2):
		super().__init__(agent_id, class_count)
		self.modality_id = 1
		self.model = tf.keras.Sequential([
			tf.keras.Input(shape=(TIMESTEPS, FEATURE_DIM)),
			layers.Masking(mask_value=0.0),
			layers.LSTM(128, return_sequences=False),
			layers.Dense(64, activation='relu'),
			layers.Dense(class_count)
		])

	def emit(self, event):
		audio = tf.convert_to_tensor(event["audio_data"], dtype=tf.float32)
		if len(audio.shape) == 1 or (len(audio.shape) > 1 and audio.shape[-1] != 1):
			audio = tf.expand_dims(audio, axis=-1) # Ensure shape is (H, W, 1)
		features = self.model(tf.expand_dims(audio, axis=0)) # Add batch dimension
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
	
	def train_step(self, x, y): 
		"""
		Train step for the audio agent.
		x: input audio tensor
		y: ground truth label
		"""
		with tf.GradientTape() as tape:
			logits = self.classifier(self.model(tf.expand_dims(x, axis=0)))
			loss = self.loss_fn(tf.convert_to_tensor([y]), logits)
		grads = tape.gradient(loss, self.model.trainable_variables + self.classifier.trainable_variables)
		self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables + self.classifier.trainable_variables))
		return float(loss.numpy())

	
class IRAgent(Agent):
	def __init__(self, agent_id, class_count=2):
		super().__init__(agent_id, class_count)
		self.modality_id = 2

	def emit(self, episode): 
		heat_signature = episode["heat_signature"]
		motion_pattern = episode["motion_pattern"]

		if heat_signature:
			if motion_pattern == "erratic":
				emergency = random.uniform(0.8, 1.0)
			else:
				emergency = random.uniform(0.4, 0.6)
		else:
			if motion_pattern == "erratic":
				emergency = random.uniform(0.2, 0.4)
			else:
				emergency = random.uniform(0.0, 0.2)

		return {
			"agent_id": self.agent_id,
			"belief": [1 - emergency, emergency],
			"prediction": np.argmax([1 - emergency, emergency]),
			"correct": (np.argmax([1 - emergency, emergency]) == episode["label"]),
			"modality_id": self.modality_id, 
			"hedge_weight": self.hedge_weight.numpy()
		}