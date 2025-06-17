import yaml
import random
import os
import csv
import threading
import logging
from agent_interface import Agent
from dataset import data_loader_for, get_validation_set
from model import model_factory

# Load configs
with open('configs/sim_config.yaml') as f:
    sim_config = yaml.safe_load(f)
with open('configs/agent_config.yaml') as f:
    agent_config = yaml.safe_load(f)

# Prepare global validation set
validation_batches = list(get_validation_set())

barrier = threading.Barrier(sim_config.get('num_agents', 1))
agents = [
    Agent(i, data_loader_for(i), model_factory, agent_config)
    for i in range(sim_config['num_agents'])
]

log_path = os.path.join('logs', 'metrics.csv')
log_lock = threading.Lock()
with open(log_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['round', 'agent_id', 'metric', 'value'])

def agent_worker(agent): 
    for round_idx in range(sim_config.get('num_rounds', 10)):
        agent.train()
        agent.compute_delta()
        peers = random.sample(
            [a for a in agents if a.agent_id != agent.agent_id],
            sim_config.get('peer_count', 2)
        )
        agent.gossip(peers)
        barrier.wait()
        agent.apply_deltas([])
        metrics = agent.evaluate(validation_batches)
        for name, val in metrics.items():
            with log_lock:
                with open(log_path, 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow([round_idx, agent.agent_id, name, val])
        barrier.wait()

threads = []
for agent in agents: 
    t = threading.Thread(target=agent_worker, args=(agent,))
    t.start()
    threads.append(t)
for t in threads: 
    t.join()

# # Prepare logging
# os.makedirs('logs', exist_ok=True)
# log_path = os.path.join('logs', 'metrics.csv')
# with open(log_path, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['round', 'agent_id', 'metric', 'value'])

#     # Simulation loop
#     for rnd in range(sim_config.get('num_rounds', 10)):
#         print(f"Round {rnd + 1}/{sim_config.get('num_rounds', 10)}")
#         # Local training and delta computation
#         for agent in agents:
#             agent.train()
#             agent.compute_delta()

#         # Gossip step
#         for agent in agents:
#             peers = random.sample(
#                 [a for a in agents if a.agent_id != agent.agent_id],
#                 sim_config.get('peer_count', 2)
#             )
#             agent.gossip(peers)

#         # Aggregation step
#         for agent in agents:
#             agent.apply_deltas([])

#         # Evaluation and logging
#         for agent in agents:
#             metrics = agent.evaluate(validation_batches)
#             for name, val in metrics.items():
#                 writer.writerow([rnd, agent.agent_id, name, val])
