import random
import numpy as np
from abc import ABC, abstractmethod
import tensorflow as tf

class Agent(ABC): 

	def __init__(self, agent_id, class_count=2, ema_alpha=0.3): 
		self.agent_id = agent_id
		self.class_count = class_count
		self.hedge_weight = tf.Variable(1.0, trainable=False)


	@abstractmethod
	def emit(self, episode):
		"""
		Single round: emit softmax + current trust.
		"""
		pass
  
class VisionAgent(Agent): 
	def __init__(self, agent_id, class_count=2): 
		super().__init__(agent_id, class_count)
		self.modality_id = 0

	def emit(self, episode): 
		weather = episode["weather"]
		lighting = episode["lighting"]
		motion_pattern = episode["motion_pattern"]

		if motion_pattern == "erratic":
			if weather == "clear" and lighting == "day":
				emergency = random.uniform(0.8, 1.0)
			else:
				emergency = random.uniform(0.4, 0.6)
		else:
			if weather == "clear" and lighting == "day":
				emergency = random.uniform(0.05, 0.35)
			else:
				emergency = random.uniform(0.2, 0.4)

		belief = [1 - emergency, emergency]
		return {
			"agent_id": self.agent_id,
			"belief": belief,
			"prediction": np.argmax(belief),
			"correct": (np.argmax(belief) == episode["label"]),
			"modality_id": self.modality_id,
			"hedge_weight": self.hedge_weight.numpy()
		}

class AudioAgent(Agent): 
	def __init__(self, agent_id, class_count=2):
		super().__init__(agent_id, class_count)
		self.modality_id = 1

	def emit(self, episode): 
		audio_clarity = episode["audio_clarity"]
		unusual_sound = episode["unusual_sound"]

		if unusual_sound:
			if audio_clarity == "high":
				emergency = random.uniform(0.7, 1.0)
			else:
				emergency = random.uniform(0.4, 0.6)
		else:
			if audio_clarity == "high":
				emergency = random.uniform(0.0, 0.2)
			else: 
				emergency = random.uniform(0.2, 0.4)
		
		return {
			"agent_id": self.agent_id,
			"belief": [1 - emergency, emergency],
			"prediction": np.argmax([1 - emergency, emergency]),
			"correct": (np.argmax([1 - emergency, emergency]) == episode["label"]),
			"modality_id": self.modality_id, 
			"hedge_weight": self.hedge_weight.numpy()
		}
	
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