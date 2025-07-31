# Swarm Project for Edge Devices. 

This is my attempt at creating a swarm learning environment. The scope of this project involve training multiple devices in swarm learning settings, using python simulation to engage 10 devices that each host a base model but run independently with local data. Training or fine-tuning will be conducted regularly to update weights with P2P gossip protocol and achieve a global model training. There are related works that utilised decentralised federated learning to achieve distributed training and is on par with the performance of centralised federated learning. In view of this, this project will mainly be focusing on using generative AI instead of simple CNN or RNN.

``` 
swarm-project/
├── README.md                   ← High-level overview, how to get started
├── models/                     ← All TFLite/Edge-TPU binaries
│   ├── base_model.tflite
│   └── edge_tpu_compiled/      ← output of edgetpu_compiler
│       └── base_model_edgetpu.tflite
├── data/                       ← Datasets & splits 
│   ├── imagenette/
│   └── splits/
│       ├── agent_00/
│       ├── agent_01/
│       └── …  
├── configs/                    ← YAML/JSON config for sim & agents
│   ├── sim_config.yaml         ← e.g. num_agents, rounds, gossip interval
│   └── agent_config.yaml       ← paths, hyper-params for local training
├── logs/                       ← runtime logs, mAP histories, weight-distances
│   └── 2025-06-xx_sim_run1/  
│       ├── agent_00.log
│       ├── agent_01.log
│       └── metrics.csv
│
├── python_sim/                 ← your Python‐only simulation environment
│   ├── requirements.txt        ← tf, torch, numpy, paho-mqtt, etc.
│   ├── sim.py                  ← orchestrates agents, rounds, gossip
│   ├── agent_interface.py      ← Agent class (train, gossip, eval)
│   ├── dataset.py              ← loaders, partition logic
│
├── cpp_agent/                  ← your C++ agent code & build files
│   ├── CMakeLists.txt          ← builds libcoral, TFLite, edgetpu delegate
│   ├── include/                ← public headers
│   │   └── Agent.hpp
│   ├── src/                    ← implementation
│   │   ├── Agent.cpp
│   │   └── main.cpp
│   └── third_party/            ← e.g., ZeroMQ, absl, Eigen submodules
│
├── scripts/                    ← helper scripts
│   ├── split.sh                ← spliting imagenette to each agent (extremely biased)
│   ├── build_cpp.sh            ← bootstrap CMake build for cpp_agent
│   ├── run_sim.sh              ← launches python_sim with config
│   └── collect_metrics.py      ← post-run aggregator/plotter
└── .gitignore
```


## 06-16
### Python simulation environement setup

Procedure: 
- Distributed 10 different classes' dataset to 10 different agents with each agent only see one class but not the other. 
- At each round, agents train with local dataset, gossip its weight deltas to 2 random peers, apply other agent's deltas to their own model weight, and validate on randomly selected data. 
- Repeat step 2 for 10 rounds. 

The results show an evident ***catastrophic forgetting***. At each round, agent adapt other overfitted model weights and worsen the validation loss over time. 

## 06-17
### Swarm learning
As yesterday was building python simulation, and today is more towards implementing the swarm feature such as social term, cognitive term, and inertia term. And here we go the tweaking. With the same hyperparameters - 10 rounds, 2 batches per round. Swarm learning performed quite poorly and is not able to consistently converge across devices. Therefore, we are going to do some hyperparameter tweaking now. 

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

## 07-31    
###

Step 1: Attention-Only Fusion Unit

What we implemented:
	•	Introduced an AttentionBlock to fuse softmax belief vectors from 9 heterogeneous agents.
	•	Input: [belief vector] + [modality type] + [agent ID].

Observation:
	•	Accuracy improved to ~86–90%.
	•	However, loss remained noisy, often spiking to 2–3 when wrong predictions were made.
	•	No discernible long-term decrease in loss; many outliers persisted.
	•	The model learned to combine agent inputs, but lacked sufficient generalisation.

Limitation:
	•	No regularization.
	•	Attention weights may have overfit to modality-specific or ID-specific patterns.
	•	No pooling mechanism → risk of overfitting to position or noisy agents.

Next step motivation:
Add pooling to promote feature abstraction and regularisation.

⸻

✅ Step 2: Attention + Global Average Pooling

What we implemented:
	•	Added GlobalAveragePooling1D after the attention output.
	•	Reduced sequence into a single feature vector before the classifier.

Observation:
	•	Loss dropped more steadily early on.
	•	Accuracy stayed around 86–90%, similar to the previous setup.
	•	Still observed spiky loss behavior when the model got confident but wrong.
	•	Pooling improved convergence smoothness but didn’t drastically improve final accuracy.

Limitation:
	•	Fusion model now more compact, but still relied on hard labels and suffered from overconfidence.

Next step motivation:
Introduce label smoothing to regularize output confidence.

⸻

✅ Step 3: Attention + Pooling + Trust Embedding + Label Smoothing

What we implemented:
	•	Switched loss function from SparseCategoricalCrossentropy to the label-smoothed version.
	•	Included trust score as an additional feature in the agent input embedding.
	•	Full input: [belief vector] + [modality type] + [agent ID] + [EMA trust score].
	•	Encouraged soft, probabilistic outputs rather than overconfident one-hot predictions.

Observation:
	•	Loss exhibited much smoother decay with fewer spikes.
	•	Accuracy climbed gradually to ~89–90%.
	•	Softmax outputs became less peaky, improving model generalisation.
	•	Better behavior under uncertainty — fewer “hard wrongs.”
	•	Incorporating trust as an input feature allowed the model to learn context-dependent weighting of agent reliability.

Limitation:
	•	Loss is still noisy.

Next step motivation:
Incorporate trust weighting to prioritize reliable agents during fusion.

⸻

✅ Step 4: Attention + Pooling + Label Smoothing + Multiplicative Trust Weighting

What we implemented:
	•	Instead of embedding trust score into input space, they are used as multiplicative weights on attention logits before softmax.
	•	Goal: Boost trustworthy agents, suppress unreliable ones.

Observation:
	•	Accuracy plateaued slightly lower (~88–89%) than embedding-based trust version.
	•	Loss became messier, with no obvious advantage over embedding trust scores.
	•	Trust weighting did not improve the model’s ability to resolve noisy inputs.
	•	Overall, marginal gains compared to cost in complexity.

Insight:
	•	Trust score might not correlate strongly with agent usefulness at a per-sample level.
	•	Embedding trust directly into the feature vector worked better, likely because it allowed the model to learn non-linear interactions between trust, modality, and belief vector.

Next step motivation:
Reassess trust strategy — perhaps combine dynamic agent masking, context-aware trust updates, or confidence calibration instead of hard multiplicative gating.