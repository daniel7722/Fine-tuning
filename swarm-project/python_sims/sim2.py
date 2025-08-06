import threading
import yaml
from agent import VisionAgent, AudioAgent
from fusion_unit import FusionUnit
import numpy as np
import argparse
import tensorflow as tf

from util.sim_load_data import load_data
from util.sim_pretrain_agent import pre_train_agents
from util.sim_log import setup_log_file

def main(filename): 

    print(tf.config.list_physical_devices("GPU"))

    # Load configs
    with open("configs/sim_config.yaml") as f:
        sim_config = yaml.safe_load(f)

    # Extract simulation parameters
    NUM_CLASSES = sim_config.get("num_classes", 2)
    NUM_AGENTS = sim_config.get("num_agents", 4)
    NUM_MODALITIES = sim_config.get("num_modalities", 2)

    # Create fusion unit
    fusion_unit = FusionUnit(
        class_count=NUM_CLASSES,
        num_agents=NUM_AGENTS,
        num_modalities=NUM_MODALITIES
    )

    # Create agents based on config
    agents = [
        VisionAgent(agent_id=0, class_count=NUM_CLASSES),
        AudioAgent(agent_id=1, class_count=NUM_CLASSES),
    ]

    # Load data
    train_data, val_data, test_data = load_data()
   
    # Pre-train only the vision agent for now
    pre_train_agents([agents[0]], train_data=train_data, val_data=val_data)

    # prompt user to continue training
    continue_training = input("Continue training? (y/n): ").strip().lower()
    if continue_training != 'y':
        print("Exiting without training.")
        return
    print("Starting training...")

    # Fusion thread barrier
    barrier = threading.Barrier(NUM_AGENTS)

    # Shared prediction list for aggregation
    emissions_lock = threading.Lock()
    emissions = []
    correct_count = 0

    # Setup logging
    log_file, csv_writer = setup_log_file(filename)
        

    def agent_worker(agent):
        nonlocal emissions
        nonlocal correct_count
        for round_idx, data in enumerate(train_data):
            gt = data["label"]
            out = agent.emit(data)
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
                        # agent_loss = agent_i.train_step(
                        #     event=data,
                        #     label=gt
                        # )
                        if agent_id == 0: 
                            manual_weight = 0.95
                        else:
                            manual_weight = 0.05
                        
                        agent_i.hedge_weight.assign(manual_weight)
                    #     updated_weight = agent_i.hedge_weight * np.exp(-eta * agent_loss)
                    #     new_weights[agent_id] = updated_weight
                    #     total_weight += updated_weight

                    # for agent_id, updated_weight in new_weights.items():
                    #     normalised_weight = updated_weight / total_weight
                    #     agent_i = next(a for a in agents if a.agent_id == agent_id)
                    #     agent_i.hedge_weight.assign(tf.clip_by_value(normalised_weight, 0.001, 1.0))
                  
                    correct_count += int(np.argmax(fused) == gt)
                    correct_percentage = correct_count / (round_idx + 1) * 100
                    csv_writer.writerow([
                        round_idx,
                        gt,
                        f"{correct_percentage:.2f}%",
                        f"{loss:.4f}",
                        np.argmax(fused),
                        [float(
                            next(a for a in agents if a.agent_id == out['agent_id']).hedge_weight.numpy()
                        ) for out in emissions]
                    ])
                    if round_idx % 100 == 0:
                        print(
                            f"Round {round_idx}: Loss: {loss:.4f}, Correct: {correct_percentage:.2f}%"
                        )
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
