import tensorflow as tf
from tensorflow.keras import layers, Model

class FusionUnit(tf.keras.Model):
    def __init__(self, class_count=2, num_agents=4, num_modalities=4, hidden_dim=16, embedding_dim=8):
        super().__init__()
        self.class_count = class_count
        self.embedding_dim = embedding_dim

        # Learned embeddings
        self.agent_embeddings = layers.Embedding(input_dim=num_agents, output_dim=embedding_dim)
        self.modality_embeddings = layers.Embedding(input_dim=num_modalities, output_dim=embedding_dim)

        # Attention network
        self.attention_net = tf.keras.Sequential([
            layers.Dense(hidden_dim, activation='relu'),
            layers.Dense(1) # Outputs a scalar attention score
        ])
        self.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def call(self, agent_outputs, training=False): 
        """
        agent_outputs: list of dicts with keys: 
          - 'belief': numpy array or tensor of shape [class_count]
          - 'trust': float scalar
          - 'agent_id': int (0-based)
          - 'modality_id': int (0-based)
        Returns: 
          - fused_output: softmax tensor of shape [class_count]
        """
        x = []
        for out in agent_outputs:             
            belief = tf.convert_to_tensor(out['belief'], dtype=tf.float32)
            trust = tf.convert_to_tensor([out['trust']], dtype=tf.float32)
            # Embed identity
            agent_id = tf.convert_to_tensor(out['agent_id'], dtype=tf.int32)
            modality_id = tf.convert_to_tensor(out['modality_id'], dtype=tf.int32)

            agent_vec = self.agent_embeddings(agent_id) # shape [1, embedding_dim]
            modality_vec = self.modality_embeddings(modality_id) # shape [1, embedding_dim]
            agent_vec = self.agent_embeddings(tf.expand_dims(agent_id, axis=0))
            modality_vec = self.modality_embeddings(tf.expand_dims(modality_id, axis=0))
            identity_vec = tf.concat([agent_vec, modality_vec], axis=-1) # shape [1, 2*embedding_dim]
            identity_vec = tf.reshape(identity_vec, [-1]) # shape [2*embedding_dim]
            combined = tf.concat([belief, trust, identity_vec], axis=0)
            x.append(combined)

        x = tf.stack(x, axis=0) # shape [num_agents, class_count + 1 + 2*embedding_dim]
        attn_scores = self.attention_net(x)
        attn_weights = tf.nn.softmax(attn_scores, axis=0)

        beliefs = tf.stack([tf.convert_to_tensor(out['belief'], dtype=tf.float32) for out in agent_outputs])
        weighted_belief = tf.reduce_sum(attn_weights * beliefs, axis=0)
        fused_output = tf.nn.softmax(weighted_belief)
        self.last_logits = weighted_belief
        return fused_output
    
    def train_on_single_example(self, agent_outputs, true_label):        
        """
        Run a training step on a single example using the provided agent_outputs and ground truth label.
        """

        with tf.GradientTape() as tape: 
            x = []
            for out in agent_outputs:             
                belief = tf.convert_to_tensor(out['belief'], dtype=tf.float32)
                trust = tf.convert_to_tensor([out['trust']], dtype=tf.float32)
                agent_id = tf.convert_to_tensor(out['agent_id'], dtype=tf.int32)
                modality_id = tf.convert_to_tensor(out['modality_id'], dtype=tf.int32)
                true_label_tensor = tf.convert_to_tensor([true_label], dtype=tf.int32)

                agent_vec = self.agent_embeddings(tf.expand_dims(agent_id, axis=0))
                modality_vec = self.modality_embeddings(tf.expand_dims(modality_id, axis=0))
                identity_vec = tf.concat([agent_vec, modality_vec], axis=-1) # shape [1, 2*embedding_dim]
                identity_vec = tf.reshape(identity_vec, [-1])
                combined = tf.concat([belief, trust, identity_vec], axis=0)
                x.append(combined)

            x = tf.stack(x, axis=0)
            attn_scores = self.attention_net(x)
            attn_weights = tf.nn.softmax(attn_scores, axis=0)
            beliefs = tf.stack([tf.convert_to_tensor(out['belief'], dtype=tf.float32) for out in agent_outputs])
            weighted_belief = tf.reduce_sum(attn_weights * beliefs, axis=0)
            logits = weighted_belief
            print(f"Logits: {logits.numpy()}, Softmax: {tf.nn.softmax(logits).numpy()}")

            loss = tf.keras.losses.sparse_categorical_crossentropy(true_label_tensor, logits, from_logits=True)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss

    