# Swarm Project for Edge Devices. 

This is my attempt at creating a swarm learning environment. The scope of this project involve training multiple devices in swarm learning settings, using python simulation to engage 10 devices that each host a base model but run independently with local data. In certain period of time, training or fine-tuning will be conducted to update weights with P2P gossip protocol and achieve a global model training. There are related works that utilised decentralised federated learning to achieve distributed training and is on par with the performance of centralised federated learning. In view of this, this project will mainly be focusing on on-device training/find-tuning with quantised models.

``` 
swarm-project/
├── README.md                   ← High-level overview, how to get started
├── models/                     ← All TFLite/Edge-TPU binaries
│   ├── base_model.tflite
│   └── edge_tpu_compiled/      ← output of edgetpu_compiler
│       └── base_model_edgetpu.tflite
├── data/                       ← Datasets & splits (COCO subsets, synthetic)
│   ├── coco_subset/
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
│   ├── utils.py                ← mAP/eval, logging helpers
│   └── notebooks/              ← exploratory Jupyter notebooks
│       └── initial_exploration.ipynb
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
│   ├── build_cpp.sh            ← bootstrap CMake build for cpp_agent
│   ├── run_sim.sh              ← launches python_sim with config
│   └── collect_metrics.py      ← post-run aggregator/plotter
└── .gitignore
```