## 07-28
### Adaptive Attention-based Swarm Fusion
It's been a while since there's been lots of reading and changing research directions. After implementing simulation for weight gossip and obsering failure due to catastophic forgetting, we pivoted toward a more biological-inspired system. The idea is to model a swarm of heterogeneous agents (e.g., vision, infrared, audio) making local predictions, and delegate the decision-making to a central lightweight fusion unit trained using attention mechanisms. 

I implemented a multi-threaded Python simulation with 9 agents emitting belief vectors and confidence scores based on their own sensor inputs (e.g., weather, audio clarity, movement). These synthetic episodes simulate an emergency detection task. The attention fusion unit consumes these belief vectors along with agant IDs and modalidty types to generate a final decision. 

Early implementation achieved ~93% accuracy --- but that was misleading due to a severly imbalanced dataset where only 6% of samples were labeled as "emergency". To address this: 
- I dded training capability to the fusion unit using cross-entropy loss. 
- I updated the synthetic data generator to include contextually coherent variables (e.g., foggy weather and high heat signature should suggest emergency). 
- The fusion unit not learns via backpropagation in an online manner. 

However, several issues remain: 
- The training data is still heavily imbalanced and causes the fusion model to plateau early. 
- Despite high accuracy, the model's confidence distribution remains conservative (e.g., output probabilities around 6o/40 split). 
- There's no historical agent performance tracking integrated into the model yet. 

Next step: rebalance the dataset or apply weighted loss, and track agent-specidfic performance to guide attention weights. 