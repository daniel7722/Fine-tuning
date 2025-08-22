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

## 08-18
### new split
[]
Pre-training agents...
Pre-training agent 0...
Epoch 1/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 16s 299ms/step - accuracy: 0.1717 - loss: 3.0395
Epoch 2/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 295ms/step - accuracy: 0.6335 - loss: 1.6233
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 296ms/step - accuracy: 0.6338 - loss: 1.6162
Epoch 3/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 309ms/step - accuracy: 0.7858 - loss: 0.9175
Epoch 4/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 297ms/step - accuracy: 0.8677 - loss: 0.6033
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 298ms/step - accuracy: 0.8670 - loss: 0.6020
Epoch 5/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 302ms/step - accuracy: 0.9257 - loss: 0.4079
Epoch 6/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 303ms/step - accuracy: 0.9557 - loss: 0.2890
Epoch 7/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 307ms/step - accuracy: 0.9654 - loss: 0.2247
Epoch 8/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 323ms/step - accuracy: 0.9731 - loss: 0.1764
31/31 ━━━━━━━━━━━━━━━━━━━━ 11s 324ms/step - accuracy: 0.9723 - loss: 0.1759
Epoch 9/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 11s 319ms/step - accuracy: 0.9805 - loss: 0.1276
Epoch 10/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 10s 304ms/step - accuracy: 0.9895 - loss: 0.1112
Saved pre-trained model for agent 0 to models/pretrained_agent_0.keras
Pre-training agent 1...
Epoch 1/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 52s 16ms/step - accuracy: 0.1589 - loss: 3.1078 - learning_rate: 0.0010
Epoch 2/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 0s 15ms/step - accuracy: 0.5080 - loss: 1.9315 - learning_rate: 0.0010
Epoch 3/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 19ms/step - accuracy: 0.6583 - loss: 1.3483 - learning_rate: 0.0010
Epoch 4/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 24ms/step - accuracy: 0.6787 - loss: 1.1944 - learning_rate: 0.0010
Epoch 5/10
30/31 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - accuracy: 0.7338 - loss: 0.9712
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 22ms/step - accuracy: 0.7328 - loss: 0.9742 - learning_rate: 0.0010
Epoch 6/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 29ms/step - accuracy: 0.7356 - loss: 0.9472 - learning_rate: 0.0010
Epoch 7/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 23ms/step - accuracy: 0.7692 - loss: 0.8480 - learning_rate: 0.0010
Epoch 8/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 26ms/step - accuracy: 0.8072 - loss: 0.7388 - learning_rate: 0.0010
Epoch 9/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 26ms/step - accuracy: 0.8186 - loss: 0.6782 - learning_rate: 0.0010
Epoch 10/10
31/31 ━━━━━━━━━━━━━━━━━━━━ 1s 27ms/step - accuracy: 0.8186 - loss: 0.6430 - learning_rate: 0.0010
Saved pre-trained model for agent 1 to models/pretrained_agent_1.keras
Pre-training complete.
Validating pre-trained agents...
7/7 ━━━━━━━━━━━━━━━━━━━━ 4s 332ms/step - accuracy: 0.6918 - loss: 0.8799
Agent 0 validation loss: 0.7878
7/7 ━━━━━━━━━━━━━━━━━━━━ 13s 2s/step - accuracy: 0.7297 - loss: 0.8916
Agent 1 validation loss: 0.9382
Continue training? (y/n): y
Starting training...
[Round 0] GT label = 7
  Agent 0 pred=7 correct=True hedge=0.0357
  Agent 1 pred=7 correct=True hedge=0.0357
[Round 1] GT label = 6
  Agent 0 pred=6 correct=True hedge=0.4988
  Agent 1 pred=6 correct=True hedge=0.5012
[Round 2] GT label = 11
  Agent 0 pred=11 correct=True hedge=0.5085
  Agent 1 pred=15 correct=False hedge=0.4915
[EVAL:val@500] n=200 | a0=0.705  a1=0.705  b0=0.725  m1=0.795
[EVAL:val@1000] n=200 | a0=0.705  a1=0.705  b0=0.705  m1=0.815
[EVAL:val_final] n=829 | a0=0.713  a1=0.697  b0=0.697  m1=0.840
[EVAL:test_final] n=829 | a0=0.688  a1=0.690  b0=0.691  m1=0.819
(fine-tuning-env) (base) danielhuang@Daniels-MacBook-Pro-2 swarm-project % python python_sims/metric/plotting.py
{'Agent0': np.float64(0.6967015285599356), 'Agent1': np.float64(0.7135961383748994), 'B0': np.float64(0.7562349155269509), 'M1': np.float64(0.7851971037811746)}
McNemar: pvalue      0.0030960205499590185
statistic   8.75
  grp = df.groupby("cos_bin")[["acc_B0","acc_M1"]].mean()
                       acc_B0    acc_M1
cos_bin                                
(0.000745, 0.06301]  0.448000  0.440000
(0.06301, 0.1777]    0.411290  0.540323
(0.1777, 0.3151]     0.524194  0.556452
(0.3151, 0.567]      0.588710  0.685484
(0.567, 0.7865]      0.768000  0.800000
(0.7865, 0.9195]     0.927419  0.927419
(0.9195, 0.9807]     0.935484  0.943548
(0.9807, 0.997]      0.975806  0.975806
(0.997, 0.9998]      0.983871  0.983871
(0.9998, 1.0]        1.000000  1.000000

### Additional metrics: 
[]
Pre-training agents...
Loaded pre-trained model for agent 0 from disk.
Loaded pre-trained model for agent 1 from disk.
Pre-training complete.
Validating pre-trained agents...
7/7 ━━━━━━━━━━━━━━━━━━━━ 5s 307ms/step - accuracy: 0.6918 - loss: 0.8799
Agent 0 validation loss: 0.7878
7/7 ━━━━━━━━━━━━━━━━━━━━ 13s 2s/step - accuracy: 0.7297 - loss: 0.8916
Agent 1 validation loss: 0.9382
Continue training? (y/n): y
Starting training...
[Round 0] GT label = 18
  Agent 0 pred=18 correct=True hedge=0.5000
  Agent 1 pred=25 correct=False hedge=0.5000
[Round 1] GT label = 2
  Agent 0 pred=2 correct=True hedge=0.5202
  Agent 1 pred=2 correct=True hedge=0.4798
[Round 2] GT label = 22
  Agent 0 pred=22 correct=True hedge=0.5244
  Agent 1 pred=22 correct=True hedge=0.4756
[EVAL:val@500::agent0] micro-acc=0.705  macro-F1=0.694
[EVAL:val@500::agent0] top confusions: [(27, 17, 4), (14, 27, 3), (6, 3, 3), (17, 14, 2), (9, 27, 2)]
[EVAL:val@500::agent1] micro-acc=0.705  macro-F1=0.684
[EVAL:val@500::agent1] top confusions: [(3, 11, 3), (9, 14, 3), (20, 11, 2), (17, 27, 2), (10, 24, 2)]
[EVAL:val@500::B0] micro-acc=0.815  macro-F1=0.797
[EVAL:val@500::B0] top confusions: [(9, 14, 4), (2, 22, 2), (20, 11, 2), (17, 27, 2), (6, 3, 2)]
[EVAL:val@500::M1] micro-acc=0.795  macro-F1=0.776
[EVAL:val@500::M1] top confusions: [(9, 14, 4), (27, 17, 3), (6, 3, 3), (20, 13, 1), (23, 3, 1)]
[EVAL:val@500] Top-5 per-class gains (acc, f1): [(15, 0.4285714285713673, 0.20606060606057897), (21, 0.09999999999998999, 0.10526315789472573), (4, 0.07142857142856629, 0.03968253968253854), (13, 0.0, -0.14285714285708861), (14, 0.0, 0.0)]
[EVAL:val@500] n=200 | a0=0.705  a1=0.705  b0=0.815  m1=0.795
2025-08-18 15:43:50.209051: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence
[EVAL:val@1000::agent0] micro-acc=0.705  macro-F1=0.694
[EVAL:val@1000::agent0] top confusions: [(27, 17, 4), (14, 27, 3), (6, 3, 3), (17, 14, 2), (9, 27, 2)]
[EVAL:val@1000::agent1] micro-acc=0.705  macro-F1=0.684
[EVAL:val@1000::agent1] top confusions: [(3, 11, 3), (9, 14, 3), (20, 11, 2), (17, 27, 2), (10, 24, 2)]
[EVAL:val@1000::B0] micro-acc=0.705  macro-F1=0.684
[EVAL:val@1000::B0] top confusions: [(3, 11, 3), (9, 14, 3), (20, 11, 2), (17, 27, 2), (10, 24, 2)]
[EVAL:val@1000::M1] micro-acc=0.810  macro-F1=0.782
[EVAL:val@1000::M1] top confusions: [(9, 14, 4), (17, 27, 2), (20, 11, 2), (6, 3, 2), (20, 13, 1)]
[EVAL:val@1000] Top-5 per-class gains (acc, f1): [(3, 0.5714285714284897, 0.39999999999989555), (24, 0.4285714285713673, 0.3179487179486875), (4, 0.35714285714283156, 0.2329192546583555), (10, 0.2499999999999687, 0.29411764705877974), (16, 0.22222222222219756, 0.16959064327484197)]
[EVAL:val@1000] n=200 | a0=0.705  a1=0.705  b0=0.705  m1=0.810
[EVAL:val_final::agent0] micro-acc=0.713  macro-F1=0.713
[EVAL:val_final::agent0] top confusions: [(1, 5, 9), (14, 27, 9), (9, 27, 9), (6, 3, 8), (27, 17, 8)]
[EVAL:val_final::agent1] micro-acc=0.697  macro-F1=0.667
[EVAL:val_final::agent1] top confusions: [(17, 27, 12), (24, 10, 9), (3, 11, 7), (5, 1, 7), (3, 6, 6)]
[EVAL:val_final::B0] micro-acc=0.697  macro-F1=0.667
[EVAL:val_final::B0] top confusions: [(17, 27, 12), (24, 10, 9), (3, 11, 7), (5, 1, 7), (3, 6, 6)]
[EVAL:val_final::M1] micro-acc=0.805  macro-F1=0.779
[EVAL:val_final::M1] top confusions: [(6, 3, 7), (5, 1, 7), (9, 14, 6), (12, 8, 4), (18, 2, 4)]
[EVAL:val_final] Top-5 per-class gains (acc, f1): [(3, 0.40540540540539444, 0.27368421052626846), (24, 0.33333333333332404, 0.16153846153846096), (13, 0.24999999999998757, 0.19367588932806035), (17, 0.24324324324323665, 0.15539906103283652), (25, 0.22727272727271697, 0.27999999999997816)]
[EVAL:val_final] n=829 | a0=0.713  a1=0.697  b0=0.697  m1=0.805
[EVAL:test_final::agent0] micro-acc=0.688  macro-F1=0.679
[EVAL:test_final::agent0] top confusions: [(6, 3, 10), (1, 5, 9), (11, 4, 7), (14, 27, 7), (9, 27, 6)]
[EVAL:test_final::agent1] micro-acc=0.690  macro-F1=0.650
[EVAL:test_final::agent1] top confusions: [(6, 3, 10), (3, 6, 10), (17, 27, 9), (20, 11, 7), (6, 11, 7)]
[EVAL:test_final::B0] micro-acc=0.691  macro-F1=0.651
[EVAL:test_final::B0] top confusions: [(6, 3, 10), (3, 6, 10), (17, 27, 9), (20, 11, 7), (6, 11, 7)]
[EVAL:test_final::M1] micro-acc=0.791  macro-F1=0.759
[EVAL:test_final::M1] top confusions: [(6, 3, 11), (20, 11, 7), (3, 6, 7), (17, 27, 6), (13, 4, 5)]
[EVAL:test_final] Top-5 per-class gains (acc, f1): [(13, 0.3999999999999801, 0.2880523731587573), (3, 0.3243243243243155, 0.22996515679442425), (24, 0.27777777777777, 0.16347687400318567), (25, 0.2727272727272604, 0.3199999999999766), (19, 0.24999999999998446, 0.07407407407406996)]
[EVAL:test_final] n=829 | a0=0.688  a1=0.690  b0=0.691  m1=0.791
(fine-tuning-env) (base) danielhuang@Daniels-MacBook-Pro-2 swarm-project % python3 python_sims/metric/plotting.py
{'Agent0': np.float64(0.6875753920386007), 'Agent1': np.float64(0.6899879372738239), 'B0': np.float64(0.6911942098914354), 'M1': np.float64(0.7913148371531966)}
McNemar: pvalue      5.6078418266289875e-14
statistic   56.50420168067227
  grp = df.groupby("cos_bin")[["acc_B0","acc_M1"]].mean()
                        acc_B0    acc_M1
cos_bin                                 
(0.0003844, 0.07219]  0.349398  0.445783
(0.07219, 0.1724]     0.385542  0.554217
(0.1724, 0.3206]      0.397590  0.650602
(0.3206, 0.556]       0.457831  0.734940
(0.556, 0.7286]       0.590361  0.843373
(0.7286, 0.8891]      0.865854  0.817073
(0.8891, 0.9758]      0.891566  0.891566
(0.9758, 0.9965]      0.975904  0.975904
(0.9965, 0.9999]      1.000000  1.000000
(0.9999, 1.0]         1.000000  1.000000
Final cum NLL: {'_cum_b0': 795.7053007633231, '_cum_m1': 660.966834425238}
ECE(M1): 0.10198587539579959


## 08-21
{'Agent0': 0.6875753920386007, 'Agent1': 0.6899879372738239, 'B0': 0.7756332931242461, 'M1': 0.8094089264173703}
McNemar: pvalue      0.0028669843168027823
statistic   8.890243902439025
Final cum NLL: {'_cum_b0': 600.2225981394434, '_cum_m1': 633.5717569018987}
ECE(M1): 0.1392944650893045
Hedge↔Attention Δ correlation: Pearson r=-0.011 (p=7.03e-01), Spearman ρ=-0.015 (p=6.09e-01)

### 08-21-1
{'Agent0': 0.6875753920386007, 'Agent1': 0.6899879372738239, 'B0': 0.7756332931242461, 'M1': 0.7696019300361882}
McNemar: pvalue      0.6906192201976447
statistic   0.15841584158415842
Final cum NLL: {'_cum_b0': 600.2225212556991, '_cum_m1': 844.6615201006782}
ECE(M1): 0.12632792573354784
Hedge↔Attention Δ correlation: Pearson r=0.083 (p=3.29e-03), Spearman ρ=0.036 (p=2.07e-01)

### 08-21-2
{'Agent0': 0.6875753920386007, 'Agent1': 0.6899879372738239, 'B0': 0.6948130277442702, 'M1': 0.758745476477684}
McNemar: pvalue      2.3193164414932735e-05
statistic   17.90728476821192
Final cum NLL: {'_cum_b0': 757.420636668031, '_cum_m1': 836.5296398979975}
ECE(M1): 0.13267875803394114
Hedge↔Attention Δ correlation: Pearson r=0.291 (p=1.29e-25), Spearman ρ=0.336 (p=3.16e-34)

### 08-21-3
{'Agent0': 0.6875753920386007, 'Agent1': 0.6899879372738239, 'B0': 0.7201447527141134, 'M1': 0.773220747889023}
McNemar: pvalue      0.0006241512131928892
statistic   11.70253164556962
Final cum NLL: {'_cum_b0': 678.7974562031374, '_cum_m1': 836.6730536339544}
ECE(M1): 0.10784973413543045
Hedge↔Attention Δ correlation: Pearson r=0.077 (p=6.86e-03), Spearman ρ=0.123 (p=1.35e-05)

### 08-21-4
{'Agent0': 0.6875753920386007, 'Agent1': 0.6899879372738239, 'B0': 0.6996381182147166, 'M1': 0.8130277442702051}
McNemar: pvalue      1.395872222478347e-14
statistic   59.23972602739726
Final cum NLL: {'_cum_b0': 744.6222009432124, '_cum_m1': 611.3911809639305}
ECE(M1): 0.1262902019340972
Hedge↔Attention Δ correlation: Pearson r=-0.183 (p=7.13e-11), Spearman ρ=-0.181 (p=1.28e-10)