import threading
import yaml
import orjson as json
from pathlib import Path
from agent import VisionAgent, AudioAgent
from fusion_unit import FusionUnit
import numpy as np
import csv
import argparse
import tensorflow as tf
import random
import pickle


def main(filename): 
    # Load configs
    with open("configs/sim_config.yaml") as f:
        sim_config = yaml.safe_load(f)
    with open("configs/agent_config.yaml") as f:
        agent_config = yaml.safe_load(f)

    # Extract simulation parameters
    NUM_CLASSES = sim_config.get("num_classes", 2)
    NUM_AGENTS = sim_config.get("num_agents", 4)
    NUM_MODALITIES = sim_config.get("num_modalities", 2)
    NUM_ROUNDS = sim_config.get("num_rounds", 20)

    # Create fusion unit
    fusion_unit = FusionUnit(
        class_count=NUM_CLASSES,
        num_agents=NUM_AGENTS,
        num_modalities=NUM_MODALITIES
    )

    # Create agents based on config
    agents = [
        VisionAgent(agent_id=0, class_count=NUM_CLASSES),
        VisionAgent(agent_id=1, class_count=NUM_CLASSES),
        VisionAgent(agent_id=2, class_count=NUM_CLASSES),
        AudioAgent(agent_id=3, class_count=NUM_CLASSES),
        AudioAgent(agent_id=4, class_count=NUM_CLASSES),
        AudioAgent(agent_id=5, class_count=NUM_CLASSES),
    ]


    # Load preprocessed AVE dataset with pickle caching
    train_path = Path("./data/AVE_Dataset/processed/train.jsonl")
    val_path = Path("./data/AVE_Dataset/processed/val.jsonl")
    test_path = Path("./data/AVE_Dataset/processed/test.jsonl")
    cache_dir = Path("./data/AVE_Dataset/processed")
    cache_train = cache_dir / "train.pkl"
    cache_val = cache_dir / "val.pkl"
    cache_test = cache_dir / "test.pkl"

    # Function to load or cache JSONL data
    def load_or_cache_jsonl(jsonl_path, pkl_path):
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                return pickle.load(f)
        else:
            with open(jsonl_path, "r") as f:
                data = [json.loads(line) for line in f]
            with open(pkl_path, "wb") as f:
                pickle.dump(data, f)
            return data

    print(f"Loading data (with cache)...")
    train_data = load_or_cache_jsonl(train_path, cache_train)
    val_data = load_or_cache_jsonl(val_path, cache_val)
    test_data = load_or_cache_jsonl(test_path, cache_test)
    print("Done loading data.")
    random.shuffle(train_data)
    
    # Fusion thread barrier
    barrier = threading.Barrier(NUM_AGENTS)

    # Shared prediction list for aggregation
    emissions_lock = threading.Lock()
    emissions = []
    correct_count = 0

    # log file setup
    log_path = Path("logs/2025-08-05")
    log_path.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path / f"AVE_{filename}.csv", "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["round", "ground_truth", "correct_percentage", "loss", "softmax_output", "hedge_weights"])
    

    def agent_worker(agent):
        nonlocal emissions
        nonlocal correct_count
        for round_idx, data in enumerate(train_data):
            gt = data["label"]
            print(f"Round {round_idx + 1}/{NUM_ROUNDS} - Agent {agent.agent_id} processing data")
            out = agent.emit(data)
            print(f"Agent {agent.agent_id} done processing data")
            with emissions_lock:
                emissions.append(out)

            # Wait for all agents to emit
            barrier.wait()  

            # Only the first agent will handle fusion and logging
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
                        agent_loss = agent_i.train_step(
                            x=np.array(pred_logits, dtype=np.float32), 
                            y=gt
                        )
                        updated_weight = agent_i.hedge_weight * np.exp(-eta * agent_loss)
                        new_weights[agent_id] = updated_weight
                        total_weight += updated_weight

                    for agent_id, updated_weight in new_weights.items():
                        normalised_weight = updated_weight / total_weight
                        agent_i = next(a for a in agents if a.agent_id == agent_id)
                        agent_i.hedge_weight.assign(tf.clip_by_value(normalised_weight, 0.001, 1.0))


                    
                    correct_count += int(np.argmax(fused) == gt)
                    correct_percentage = correct_count / (round_idx + 1) * 100
                    print(f"Round {round_idx}: Loss: {loss:.4f}, Correct: {correct_percentage:.2f}%")
                    csv_writer.writerow([
                        round_idx, 
                        gt,
                        f"{correct_percentage:.2f}%",
                        f"{loss:.4f}",
                        fused.numpy().tolist(),
                        [float(next(a for a in agents if a.agent_id == out['agent_id']).hedge_weight.numpy()) for out in emissions]
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
