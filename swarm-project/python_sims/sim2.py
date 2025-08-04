import threading 
import yaml
import json
from pathlib import Path
from agent import VisionAgent, AudioAgent, IRAgent
from fusion_unit import FusionUnit
import numpy as np
import csv
import argparse
import tensorflow as tf

def main(filename): 
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
    log_path = Path("logs/2025-07-31")
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path / f"emergency_sim_{filename}.csv", "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["round", "ground_truth", "emergency_percentage", "correct_percentage", "loss", "softmax_output", "hedge_weights"])
    emissions = []
    correct_count = 0
    emergency_count = 0
    non_emergency_count = 0

    def agent_worker(agent):
        nonlocal emissions
        nonlocal correct_count
        nonlocal emergency_count
        nonlocal non_emergency_count
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

                    eta = 0.5
                    total_weight = 0.0
                    new_weights = {}

                    # Update hedge weights based on correctness
                    for out in emissions: 
                        agent_id = out['agent_id']
                        agent_i = next(a for a in agents if a.agent_id == agent_id)
                        pred_logits = out['belief']
                        agent_loss = tf.keras.losses.sparse_categorical_crossentropy(
                            tf.convert_to_tensor([gt], dtype=tf.int32), 
                            tf.convert_to_tensor([pred_logits], dtype=tf.float32), 
                            from_logits=False
                        ).numpy()[0]
                        updated_weight = agent_i.hedge_weight * np.exp(-eta * agent_loss)
                        new_weights[agent_id] = updated_weight
                        total_weight += updated_weight

                    for agent_id, updated_weight in new_weights.items():
                        normalised_weight = updated_weight / total_weight
                        agent_i = next(a for a in agents if a.agent_id == agent_id)
                        agent_i.hedge_weight.assign(tf.clip_by_value(normalised_weight, 0.001, 1.0))


                    
                    correct_count += int(np.argmax(fused) == gt)
                    if gt == 1:
                        emergency_count += 1
                    else: 
                        non_emergency_count += 1
                    
                    emergency_percentage = emergency_count / (round_idx + 1) * 100
                    correct_percentage = correct_count / (round_idx + 1) * 100
                    print(f"Round {round_idx}: Loss: {loss:.4f}, Correct: {correct_percentage:.2f}%")
                    csv_writer.writerow([
                        round_idx, 
                        gt,
                        f"{emergency_percentage:.2f}%",
                        f"{correct_percentage:.2f}%",
                        f"{loss:.4f}",
                        fused.numpy().tolist(),
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

if __name__ == "__main__": 
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="Name for log file for this run")
    args = parser.parse_args()
    if not args.name:
        raise ValueError("Run name cannot be empty")
    run_name = args.name.strip()
    main(run_name)
