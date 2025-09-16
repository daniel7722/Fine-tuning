## 06-16
### Python simulation environement setup

Procedure: 
- Distributed 10 different classes' dataset to 10 different agents with each agent only see one class but not the other. 
- At each round, agents train with local dataset, gossip its weight deltas to 2 random peers, apply other agent's deltas to their own model weight, and validate on randomly selected data. 
- Repeat step 2 for 10 rounds. 

The results show an evident ***catastrophic forgetting***. At each round, agent adapt other overfitted model weights and worsen the validation loss over time. 