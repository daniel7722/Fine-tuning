import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention

class FusionUnit(tf.keras.Model):
    def __init__(self, class_count=2, num_agents=4, num_modalities=4, hidden_dim=16, embedding_dim=8):
        super().__init__()
        self.class_count = class_count
        self.embedding_dim = embedding_dim # Dimension for agent and modality embeddings
        # hidden_dim: Dimension for the internal latent representation in the attention block

        # Learned embeddings
        self.agent_embeddings = layers.Embedding(input_dim=num_agents, output_dim=embedding_dim)
        self.modality_embeddings = layers.Embedding(input_dim=num_modalities, output_dim=embedding_dim)

        # Attention network
        self.input_proj = layers.Dense(hidden_dim)
        self.attn_norm1 = LayerNormalization()
        self.attn_block = MultiHeadAttention(num_heads=2, key_dim=embedding_dim, dropout=0.1)
        self.attn_norm2 = LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(hidden_dim * 2, activation='relu'), 
            layers.Dropout(0.1),  # Dropout for regularization
            layers.Dense(hidden_dim)
        ])
        self.output_layer = layers.Dense(class_count)
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0) # Clip gradients to prevent exploding gradients
        self.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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

            agent_vec = self.agent_embeddings(tf.expand_dims(agent_id, axis=0))
            modality_vec = self.modality_embeddings(tf.expand_dims(modality_id, axis=0))
            identity_vec = tf.concat([agent_vec, modality_vec], axis=-1) # shape [1, 2*embedding_dim]
            identity_vec = tf.reshape(identity_vec, [-1]) # shape [2*embedding_dim]
            combined = tf.concat([belief, trust, identity_vec], axis=0)
            x.append(combined)

        x = tf.stack(x, axis=0) # shape [num_agents, 2 (belief vector) + 1 (trust) + 2*embedding_dim]
        x = tf.expand_dims(x, axis=0)  # [1 (batch_num), num_agents, features]

        # Project to hidden_dim first
        x_proj = self.input_proj(x) # [1 (batch_num), num_agents, hidden_dim]

        # Attention Block (self-attention)
        attn_out = self.attn_block(query=x_proj, key=x_proj, value=x_proj)
        attn_out = self.attn_norm1(attn_out + x_proj)  # Residual connection + norm

        # Feed-forward network
        ffn_out = self.ffn(attn_out) 
        ffn_out = self.attn_norm2(ffn_out + attn_out)  # Residual connection + norm

        ffn_out = tf.squeeze(ffn_out, axis=0)  # [num_agents, hidden_dim] removing batch dimension

        attn_logits = self.output_layer(ffn_out) # [num_agents, class_count]
        agent_weights = tf.nn.softmax(layers.Dense(1)(ffn_out), axis=0)
        aggregated_logits = tf.reduce_mean(attn_logits * agent_weights, axis=0)  # [class_count]

        self.last_logits = aggregated_logits
        fused_output = tf.nn.softmax(aggregated_logits)
        return fused_output
    
    def train_on_single_example(self, agent_outputs, true_label):
        with tf.GradientTape() as tape:
            fused_output = self.call(agent_outputs, training=True)
            true_label_tensor = tf.convert_to_tensor([true_label], dtype=tf.int32)
            loss = tf.keras.losses.sparse_categorical_crossentropy(true_label_tensor, self.last_logits, from_logits=True)

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return loss.numpy().item()

    