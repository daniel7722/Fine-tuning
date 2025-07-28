import threading 
import yaml
import random
import time
from agent import Agent
from fusion_unit import FusionUnit
import numpy as np


# Load configs
with open("configs/sim_config.yaml") as f:
    sim_config = yaml.safe_load(f)
with open("configs/agent_config.yaml") as f:
    agent_config = yaml.safe_load(f)

NUM_CLASSES = sim_config.get("num_classes", 2)
NUM_AGENTS = sim_config.get("num_agents", 4)
NUM_MODALITIES = sim_config.get("num_modalities", 2)
NUM_ROUNDS = sim_config.get("num_rounds", 20)

fusion_unit = FusionUnit(
    class_count=NUM_CLASSES,
    num_agents=NUM_AGENTS,
    num_modalities=NUM_MODALITIES
)

def get_ground_truth(): 
    return random.randint(0, NUM_CLASSES - 1)

agents = [
    Agent(agent_id=i, class_count=NUM_CLASSES)
    for i in range(NUM_AGENTS)
]

for i, agent in enumerate(agents): 
    agent.modality_id = i % NUM_MODALITIES  # Assign modalities in a round-robin fashion for now

# Fusion thread barrier
barrier = threading.Barrier(NUM_AGENTS)

# Shared prediction list for aggregation
emissions_lock = threading.Lock()
emissions = []

def agent_worker(agent): 
    global emissions
    for round_idx in range(NUM_ROUNDS): 
        gt = get_ground_truth()
        out = agent.emit(gt)
        out["modality_id"] = agent.modality_id  # Add modality ID
        with emissions_lock:
            emissions.append(out)

        barrier.wait()  # Wait for all agents to emit
        if agent.agent_id == 0:
            with emissions_lock:
                # Aggregate outputs from all agents
                fused = fusion_unit.call(agent_outputs=emissions)
                print(f"Round {round_idx}, Fused Output: {fused.numpy()}")
                emissions.clear()

        barrier.wait()  # Wait for fusion to complete before next round

threads = []
for agent in agents: 
    t = threading.Thread(target=agent_worker, args=(agent,))
    t.start()
    threads.append(t)

for t in threads:
  t.join()