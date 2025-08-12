import threading
from agent import VisionAgent, AudioAgent
from fusion_unit import FusionUnit
import numpy as np
import argparse
import tensorflow as tf
import yaml

from util.sim_load_data import load_data
from util.sim_pretrain_agent import pre_train_agents
from util.sim_log import setup_log_file
from util.sim_update_hedge import update_hedge

def main(filename, pretrain): 

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
    pre_train_agents(agents, train_data, val_data, pretrain)

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

        def safe_barrier_wait(): 
            try:
                barrier.wait()
                return True
            except threading.BrokenBarrierError:
                return False
            
        try: 
            for round_idx, data in enumerate(train_data.take(sim_config.get("max_rounds", 5000))):
                gt = int(data["label"].numpy())
                out = agent.emit({
                    "vision_data": data["vision_data"].numpy(),
                    "audio_waveform": data["audio_waveform"].numpy(), 
                    "label": gt
                })
                with emissions_lock:
                    emissions.append(out)

                # Wait for all agents to emit
                barrier.wait()

                # Only the first agent will handle fusion and logging
                if agent.agent_id == 0:
                    with emissions_lock:
                        hedge_cfg = sim_config.get("hedge_weights", {})
                        update_hedge(hedge_cfg, fusion_unit, emissions, gt, agents, round_idx, csv_writer)
                        emissions.clear()
                if not safe_barrier_wait():
                    return
        finally:
            try: 
                barrier.abort()
            except Exception:
                pass
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
    parser.add_argument("-p", "--pretrain", action="store_true", help="Whether to pretrain agents")
    args = parser.parse_args()
    if not args.name:
        raise ValueError("Run name cannot be empty")
    run_name = args.name.strip()
    pretrain = args.pretrain
    main(run_name, pretrain=pretrain)
