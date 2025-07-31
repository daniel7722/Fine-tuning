import threading 
import yaml
import random
import time
import json
from pathlib import Path
from agent import Agent, VisionAgent, AudioAgent, IRAgent
from fusion_unit import FusionUnit
import numpy as np
import csv


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

agents = [
    VisionAgent(agent_id=0, class_count=NUM_CLASSES),
    VisionAgent(agent_id=1, class_count=NUM_CLASSES),
    VisionAgent(agent_id=2, class_count=NUM_CLASSES),
    AudioAgent(agent_id=3, class_count=NUM_CLASSES),
    AudioAgent(agent_id=4, class_count=NUM_CLASSES),
    AudioAgent(agent_id=5, class_count=NUM_CLASSES),
    IRAgent(agent_id=6, class_count=NUM_CLASSES),
    IRAgent(agent_id=7, class_count=NUM_CLASSES),
    IRAgent(agent_id=8, class_count=NUM_CLASSES),
]

# Load synthetic dataset
DATA_PATH = Path("data/synthetic/emergency_data.jsonl")
with open(DATA_PATH, "r") as f:
    episodes = [json.loads(line) for line in f]

assert len(episodes) >= NUM_ROUNDS, "Not enough data to simulate"

# Fusion thread barrier
barrier = threading.Barrier(NUM_AGENTS)

# Shared prediction list for aggregation
emissions_lock = threading.Lock()
emissions = []
correct_count = 0
emergency_count = 0
non_emergency_count = 0
log_path = Path("logs/2025-07-30")
log_path.mkdir(parents=True, exist_ok=True)
log_file = open(log_path / "emergency_sim_attention_pooling.csv", "w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["round", "ground_truth", "emergency_percentage", "correct_percentage", "loss", "softmax_output"])

def agent_worker(agent): 
    global emissions
    global correct_count
    global emergency_count
    global non_emergency_count
    for round_idx in range(NUM_ROUNDS):
        gt = episodes[round_idx]["label"]
        out = agent.emit(episodes[round_idx])
        with emissions_lock:
            emissions.append(out)

        barrier.wait()  # Wait for all agents to emit
        if agent.agent_id == 0:
            with emissions_lock:
                # Aggregate outputs from all agents
                fused = fusion_unit.call(agent_outputs=emissions)
                loss = fusion_unit.train_on_single_example(emissions, true_label=gt)
                
                print(f"Round {round_idx}")
                print(f"Ground Truth: {gt}, Loss: {loss:.4f}")
                correct_count += int(np.argmax(fused) == gt)
                if gt == 1:
                    emergency_count += 1
                else: 
                    non_emergency_count += 1
                  
                emergency_percentage = emergency_count / (round_idx + 1) * 100
                correct_percentage = correct_count / (round_idx + 1) * 100
                print(
                    f"Emergency Percentage: {emergency_percentage:.2f}%, "
                    f"Correct Percentage: {correct_percentage:.2f}%"
                )
                csv_writer.writerow([
                    round_idx, 
                    gt,
                    f"{emergency_percentage:.2f}%",
                    f"{correct_percentage:.2f}%",
                    f"{loss:.4f}",
                    fused.numpy().tolist()
                ])
                emissions.clear()

        barrier.wait()  # Wait for fusion to complete before next round

threads = []
for agent in agents: 
    t = threading.Thread(target=agent_worker, args=(agent,))
    t.start()
    threads.append(t)

for t in threads:
  t.join()
log_file.close()