import random
import numpy as np

class Agent: 

  def __init__(self, agent_id, class_count=2, ema_alpha=0.3): 
    self.agent_id = agent_id
    self.modality_id = 0  # Default modality ID, can be set later
    self.class_count = class_count
    self.ema_alpha = ema_alpha
    self.trust_score = 1.0 # Start with full trust
    self.last_correct = True # Initial assumption
    self.noise_level = random.uniform(0.0, 0.5) # Simulate sensor quality

  def predict(self, ground_truth): 
    """
    Simulates a softmax belief vector based on agent's noise level.
    The more noise, the more likely the prediction is incorrect. 
    """
    correct_class = ground_truth
    if (random.random() < self.noise_level): 
      # Predict wrong class with low confidence
      wrong_class = (correct_class + 1) % self.class_count
      probs = np.zeros(self.class_count)
      probs[wrong_class] = random.uniform(0.5, 0.6)  # Low confidence in wrong class
      probs[correct_class] = 1 - probs[wrong_class]  # Remaining confidence in correct class
    else:
      # Predict correct class with high confidence
      probs = np.zeros(self.class_count)
      probs[correct_class] = random.uniform(0.7, 1.0)
      probs[(correct_class + 1) % self.class_count] = 1 - probs[correct_class]
    return probs
  
  def update_trust(self, predicted_class, ground_truth): 
    """
    Updates trust score based on prediction correctness using EMA.
    """
    correct = (predicted_class == ground_truth)
    self.last_correct = correct
    self.trust_score = self.ema_alpha * float(correct) + (1 - self.ema_alpha) * self.trust_score

  def emit(self, ground_truth):
    """
    Single round: emit softmax + current trust.
    """
    probs = self.predict(ground_truth)
    predicted_class = np.argmax(probs)
    self.update_trust(predicted_class, ground_truth)
    return {
      "agent_id": self.agent_id,
      "belief": probs,
      "trust": self.trust_score,
      "prediction": predicted_class,
      "correct": (predicted_class == ground_truth),
      "modality_id": self.modality_id
    }