import pickle
from pathlib import Path

def pre_train_agents(agents, train_data, val_data):
        """
        Pre-train agents on the training set.
        """
        print("Pre-training agents...")
        
        for agent in agents: 
            if not Path(f"models/{agent.agent_id}.pkl").exists():
                agent.pretrain(train_data=train_data, batch_size=32, epochs=5)
                with open (f"models/{agent.agent_id}.pkl", "wb") as f:
                    pickle.dump(agent.model, f)
            else: 
                agent.model = pickle.load(open(f"models/{agent.agent_id}.pkl", "rb"))
        print("Pre-training complete.")

        print("Validating pre-trained agents...")
        for agent in agents:
            val_loss = agent.validate_pretraining(val_data=val_data, batch_size=32)
            print(f"Agent {agent.agent_id} validation loss: {val_loss:.4f}")