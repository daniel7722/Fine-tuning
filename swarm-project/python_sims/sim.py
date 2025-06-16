import yaml
import random
import os
import csv
from agent_interface import Agent
from dataset import data_loader_for, get_validation_loader, get_shuffled_validation_loader
from model import model_factory

# from dataset import data_loader_for
loader = data_loader_for(0)
imgs, labels = next(loader)
print(imgs.shape, labels.shape)  # e.g. (batch_size, H, W, C), (batch_size,)

# Load configs
with open('configs/sim_config.yaml') as f:
    sim_config = yaml.safe_load(f)
with open('configs/agent_config.yaml') as f:
    agent_config = yaml.safe_load(f)

# Prepare validation loader
validation_loader = get_shuffled_validation_loader()

# Instantiate agents
agents = [
    Agent(i, data_loader_for(i), model_factory, agent_config)
    for i in range(sim_config['num_agents'])
]

# Prepare logging
os.makedirs('logs', exist_ok=True)
log_path = os.path.join('logs', 'metrics.csv')
with open(log_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['round', 'agent_id', 'metric', 'value'])

    # Simulation loop
    for rnd in range(sim_config.get('num_rounds', 10)):
        # Local training and delta computation
        for agent in agents:
            agent.train()
            agent.compute_delta()

        # Gossip step
        for agent in agents:
            peers = random.sample(
                [a for a in agents if a.agent_id != agent.agent_id],
                sim_config.get('peer_count', 2)
            )
            agent.gossip(peers)

        # Aggregation step
        for agent in agents:
            agent.apply_deltas([])

        # Evaluation and logging
        for agent in agents:
            metrics = agent.evaluate(validation_loader)
            for name, val in metrics.items():
                writer.writerow([rnd, agent.agent_id, name, val])
