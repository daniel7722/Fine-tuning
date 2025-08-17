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

Step 5: Attention + Label Smoothing + Hedge Attention Fusion
What we implemented: 
We replaced both the EMA-based trust score and multiplicative weighting scheme with a pricipled online learning appraoch inspired by the Hedge algorithm. This formulation brings theoretical guarantees under adversarial conditions and is well-suited for dynamically learning to weigh multiple experts. 

Mathematical Formulation
Let there be N agents indexed by i \in \{1, \dots, N\}, each emitting a belief vector \mathbf{p}_i \in \mathbb{R}^C over C classes.

1. Hedge weight initialization

Each agent starts with an equal exponential weight:
w_i^{(0)} = 1, \quad \forall i

2. Attention-based fusion

Let \alpha_i^{(t)} be the attention logit score from the attention network at time t. Before softmax, we apply hedge weighting:

\tilde{\alpha}_i^{(t)} = \log w_i^{(t)} + \alpha_i^{(t)}
\text{Attention weight: } a_i^{(t)} = \frac{\exp(\tilde{\alpha}_i^{(t)})}{\sum_j \exp(\tilde{\alpha}_j^{(t)})}

This biases the softmax attention toward historically better-performing agents.

3. Loss and prediction

The fusion prediction is:
\hat{\mathbf{y}}^{(t)} = \sum_i a_i^{(t)} \cdot \mathbf{p}_i
We use label-smoothed cross-entropy loss against the ground truth.

4. Hedge weight update

For each agent i, we measure correctness of its individual prediction:
\ell_i^{(t)} = \mathbb{1}[ \arg\max \mathbf{p}_i \neq y^{(t)} ]
Then update hedge weights multiplicatively:
w_i^{(t+1)} = w_i^{(t)} \cdot \gamma^{\ell_i^{(t)}}
\quad \text{with } \gamma < 1
This punishes incorrect predictions while keeping weights normalized.

The reason we use hedge algorithm is because it's a theoritecal proven that minimised regret compared to the best expert in hindsight. Also, unlike EMA, Hedge accumulates error multiplicatively, encouraging long-term reliability. Besides it's not a learnable weights and it naturally downweights noisy or misbehaving agent. 

Emprical Results
	•	Final accuracy: 87%, slightly lower than trust embedding (89%)
	•	Loss remained noisy with many large spikes (up to 6-7)
    •	Performance plateaued earlier, suggesting limited adaptability to dynamically shifting agent reliability. 

Why might it underperform
	•	The synthtic dataset lacks sufficient heterogeneity or difficulty to differentiate reliable from unreliable agents.
	•	Hedge is sensiive to binary correctness; it ignores how wrong the prediction was (no graded penalty). 
    •	Embedding trust into the attention unit allows for nonlinear correlations, which Hedge can't model

✅ Step 6: Attention + Label Smoothing + Hedge Weight Embedding + Adaptive Hedge Update

What we implemented:
•	Embedded hedge weights (log-scaled) into the agent input representation, rather than using them directly to modify attention logits.
•	Each agent’s hedge weight is appended to its embedding vector alongside belief, agent ID, and modality ID.
•	This allows the attention mechanism to learn how to use agent reliability in a non-linear, context-sensitive way.
•	Introduced adaptive hedge update: weights are updated based on a regret-style penalty that increases if the agent repeatedly fails to predict the correct class.

Hedge Update (Adaptive Regret-Based):
Let y be the ground-truth label and p_i the belief vector of agent i.
We compute the per-agent loss as:
    l_i = -log(p_i[y])  (negative log likelihood of true class)
Then update weights:
    w_i ← w_i * γ^l_i     where γ ∈ (0,1)

Observation:
•	Final accuracy ~89%, comparable to earlier trust embedding methods.
•	Loss remained noisy despite label smoothing and EMA smoothing in plots.
•	Slower convergence and more volatility compared to simple attention-only model.

Why it might still help:
•	Embedding hedge allows richer interactions than simple gating.
•	Regret-based updates respond to probabilistic errors, not just hard mistakes.
•	Learns from context even if emissions are unordered due to multi-threading.

Limitation:
•	Still heavily bottlenecked by synthetic data's lack of diversity and realism.
•	Performance differences remain marginal across trust schemes due to low task complexity.

Next step motivation:
Move to a real-world multimodal dataset and richer expert models to test robustness, scalability, and practical gains of adaptive trust fusion.


## 08-06
I try just plop the whole foreign dataset to the system, hoping both models will learn something during the process and become a better model, and fusion unit, though will experience some hard time during the beginning, will pick up when agents start to produce good result. But turns out, the final result suck and there are several reasons that cause this. For a primary reason, the agents never learn good features. They are producing random result the whole time, making fusion unit really hard to understand what's going on. Therefore, I decided to first produce a good agent, then we can put verdict on the performance of the fusion unit. 

Real dataset of AVE is implemented with vision agent having efficientnetb0 as its based and LSTM based model for audio agent. Vision is pretrained with imagenet while audio agent has a randomly initialised model. Data distribution is a three-fold split with training/validating/testing datasets with 6/2/2 split. The setup is like we pre-train both models with training dataset in 5 epoch. Then, use validationg set to see check the quality of the resulting pretrained models. Finally, using testing data to train fusion unit to see if it has learned anything. 

So far the result is very demotivating. We have good pretrained accuracy to 0.93 for vision model. Reasonable since it has pre-trained weight on imagenet. But bad validation loss implies a significant overfit, leading to a not so ideal model. Audio model learns nothing as expected since it's random initialisation. Hence, from this pov, I was expecting maybe fusion unit would trust vision agent fully and make a good result of that. But it turns out fusion unit learns nothing again with really bad result. 

Hence, the next run name "att_2" will be adapting to this situation where we 
- freeze vision model backbone and only train the classifier to avoid overfitting
- don't pretrain audio model to make it a fully randomised model
- manually assign hedge weight so I know vision model is always more trusted than audio model
- freeze the continual improvement as well during sim
- log fusion loss every 100 rounds

Cool, this run is at least going somewhere. Here's the output:
``` 
[]
Loading data (with cache)...
Done loading data.
Data loaded in 344.77 seconds
Pre-training agents...
Epoch 1/5
78/78 ━━━━━━━━━━━━━━━━━━━━ 388s 338ms/step - accuracy: 0.3383 - loss: 2.6270  
Epoch 2/5
78/78 ━━━━━━━━━━━━━━━━━━━━ 26s 337ms/step - accuracy: 0.7436 - loss: 0.9275
Epoch 3/5
78/78 ━━━━━━━━━━━━━━━━━━━━ 25s 316ms/step - accuracy: 0.8103 - loss: 0.6387
Epoch 4/5
78/78 ━━━━━━━━━━━━━━━━━━━━ 24s 303ms/step - accuracy: 0.8579 - loss: 0.4873
Epoch 5/5
78/78 ━━━━━━━━━━━━━━━━━━━━ 24s 305ms/step - accuracy: 0.8958 - loss: 0.3693
Pre-training complete.
Validating pre-trained agents...
Agent 0 validation loss: 0.9069
...
Round 2400: Loss: 0.0293, Correct: 75.43%
```

By overcoming overfitting for vision agent, leaving audio agent random, and leverage that with hard-coded hedge weights that always trust vision more, there's accuracy improvement with training data. Although, vision always produce accurate result, the fusion accuracy is at quite low 75%, which is great improvement from random 4% but it is still affected by audio agent, which is expected. But the level of influence has no quantifiable way to explain. That is, fixing hedge weight at 0.95 vs 0.05 will always leave some level of trust towards audio agent, an amount that cannot explained by the difference between fully trusting vision agent, which accuracy is roughly 0.9 but strictly speaking unknown, and 95% trusting. This is still area to dig deeper.


## 08-07 (Agent Pretraining Improvements & Error Fixes)
- **Audio agent pretraining**:  
  - Adjusted `AudioAgent` input shape & LSTM handling to prevent `ValueError` from Dense expecting (batch,64) but receiving (batch,time,128).  
  - Fixed embedding extraction and ensured padded batching shape `[None, 128]`.
- **Vision agent unchanged**, still using frozen backbone + trainable classifier.
- **Verified pretraining loop**:  
  - Both agents now run pretraining without shape errors.  
  - Audio agent still improves marginally each epoch (slow learning rate, stable).
- **Threading barrier safety**:  
  - Added logic to avoid deadlocks when one thread finishes early.  
  - Made iterator handling safe for `OUT_OF_RANGE` so all threads exit cleanly.

## 08-08 (Full Pretraining + Fusion Unit Run)
- **Full run with ~2.5k rounds**:  
  - Pretrained both agents successfully (Vision ~0.83 acc final, Audio slow but improving).  
  - Ran fusion unit with fixed hedge 0.95 / 0.05.  
  - Achieved smooth loss curve & steady accuracy climb to ~83% final.  
  - No premature stop — confirmed earlier “900 rounds” issue was due to dataset iterator limit in config.
- **Plotted metrics**:  
  - Loss vs. round: clean downward trend.  
  - Accuracy vs. round: steady upward climb, no major instability.
- **Next step planning**:  
  - Re-enable adaptive hedge update with:  
    - Small η for stability.  
    - Clipping & renormalisation.  
    - Optional temperature scaling in fusion logits.  
  - Expect loss curve to get noisier due to dynamic weighting.
  
## 08-12
  1. Fixing Audio Model Input & Training Pipeline
	•	Discovered that the AudioAgent was throwing shape errors (Dense expecting (batch, 64) but getting (batch, time, 128)).
	•	Adjusted input shape & LSTM handling so the recurrent output was pooled/flattened before passing to Dense layers.
	•	Ensured padded batching produced shape [None, 128] and worked consistently in both training and evaluation.

⸻

2. Verifying Pretraining Works End-to-End
	•	Ran pretraining loop for both agents:
	•	Vision Agent: Frozen backbone (EfficientNetB0 pretrained on ImageNet) + trainable classifier to avoid overfitting.
	•	Audio Agent: Random initialization, learning from scratch.
	•	Confirmed audio agent improved marginally per epoch (slow but stable learning).
	•	Vision agent maintained strong accuracy without rapid overfitting.

⸻

3. Handling Thread Safety & Iterator Issues
	•	Fixed threading barriers to prevent deadlocks when one thread finished early.
	•	Added safe handling for OUT_OF_RANGE exceptions so all threads exit cleanly without hanging.
	•	Resolved earlier issue where the simulation stopped prematurely (~900 rounds) by fixing dataset iterator limits.

⸻

4. Ensuring Audio Data Integrity
	•	Diagnosed that TFRecord audio pipeline was outputting constant -1.0s due to incorrect decoding/normalization.
	•	Compared outputs of TFRecord, MoviePy, and librosa:
	•	Only librosa’s clip was valid initially.
	•	Patched TFRecord reading to normalize audio globally in sim_load_data.py.
	•	Verified TFRecord and librosa now match closely (MAE ≈ 1.8e-5, corr ≈ 1.0).

⸻

5. Full Simulation with Fixed Hedge
	•	Used fixed hedge weights: Vision = 0.95, Audio = 0.05 (to ensure vision dominance given audio’s weakness).
	•	Froze both agents’ weights during fusion to isolate fusion performance.
	•	Logged fusion loss every 100 rounds for clearer trend monitoring.

⸻

6. Results from Full 2.5k-Round Run
	•	Vision agent alone: ~0.83 accuracy final (pretrained classifier only).
	•	Audio agent: low accuracy, as expected, but stable.
	•	Fusion unit: smooth downward loss curve, steady accuracy climb to ~83%.
	•	Achieved stability — no dataset or threading errors.

⸻

7. Next Steps Identified
	•	Gradually reintroduce adaptive hedge:
	•	Use small η (learning rate) for stability.
	•	Add weight clipping & renormalization.
	•	Optionally add temperature scaling to fusion logits for calibration.
	•	Longer term: test with richer datasets & more balanced audio model so fusion can meaningfully outperform vision-only.

### Plotting
# Heatmap for the final round
python -m python_sims.util.plotting \
  --log_dir logs/2025-08-12 \
  --perclass_csv AVE_perclass.csv \
  --heatmap out/heatmap_last.png

# Heatmap for a specific round
python -m python_sims.util.plotting \
  --log_dir logs/2025-08-12 \
  --perclass_csv AVE_perclass.csv \
  --round 1200 \
  --heatmap out/heatmap_r1200.png

# Accuracy/Loss curves
python -m python_sims.util.plotting \
  --log_dir logs/2025-08-12 \
  --main_csv AVE_run.csv \
  --curves out/curves.png

# Hedge weights
python -m python_sims.util.plotting \
  --log_dir logs/2025-08-12 \
  --main_csv AVE_run.csv \
  --hedge out/hedge.png

# Per-class curve over time (e.g., class 7)
python -m python_sims.util.plotting \
  --log_dir logs/2025-08-12 \
  --perclass_csv AVE_perclass.csv \
  --class_curve 7 out/class7_curve.png