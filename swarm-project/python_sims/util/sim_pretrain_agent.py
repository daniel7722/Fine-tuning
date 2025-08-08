from pathlib import Path
import tensorflow as tf

def pre_train_agents(agents, train_data, val_data):
        """
        Pre-train agents on the training set.
        """
        print("Pre-training agents...")

        
        for agent in agents: 
            model_path = Path(f"./models/pretrained_agent_{agent.agent_id}.keras")
            if not model_path.exists():
                print(f"Pre-training agent {agent.agent_id}...")
                agent.pretrain(train_data=train_data)
                
                # Save right after pretraining
                agent.model.save(model_path)
                print(f"Saved pre-trained model for agent {agent.agent_id} to {model_path}")
            else:
                agent.model = tf.keras.models.load_model(model_path)
                agent.model.compile(optimizer=agent.optimizer, loss=agent.loss_fn, metrics=["accuracy"])
                agent.backbone=agent.model.layers[0]
                agent.classifier=agent.model.layers[1]
                print(f"Loaded pre-trained model for agent {agent.agent_id} from disk.")
        print("Pre-training complete.")

        print("Validating pre-trained agents...")
        for agent in agents:
            val_loss = agent.validate_pretraining(val_data=val_data)
            print(f"Agent {agent.agent_id} validation loss: {val_loss:.4f}")