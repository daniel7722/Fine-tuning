import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention
from util.sim_update_hedge import mix_beliefs_hedge

class FusionUnit(tf.keras.Model):
    def __init__(self, class_count=2, num_agents=4, num_modalities=4, hidden_dim=16, embedding_dim=8, lambda_poe=0.5, epsilon=1e-12, use_hedge_feat=True):
        super().__init__()
        # attention parameters
        self.class_count = class_count
        self.embedding_dim = embedding_dim # Dimension for agent and modality embeddings 

        # PoE parameters
        self.lambda_poe = tf.Variable(lambda_poe, trainable=False, dtype=tf.float32, name="lambda_poe")
        self.epsilon = float(epsilon)
        self.use_hedge_feat = use_hedge_feat

        # Hedge parameters
        self.last_pi = None          # shape [N]
        self.last_o = None           # shape [N, K]
        self.last_r = None           # shape [K]
        self.last_p_hedge = None     # shape [K]
        self.last_fused = None       # shape [K]

        # Learned embeddings
        self.agent_embeddings = layers.Embedding(input_dim=num_agents, output_dim=embedding_dim)
        self.modality_embeddings = layers.Embedding(input_dim=num_modalities, output_dim=embedding_dim)

        # Attention network
        self.input_proj = layers.Dense(hidden_dim) # Dimension for the internal latent representation in the attention block
        self.attn_norm1 = LayerNormalization()
        self.attn_block = MultiHeadAttention(num_heads=2, key_dim=embedding_dim, dropout=0.1)
        self.attn_norm2 = LayerNormalization()
        self.ffn = tf.keras.Sequential([
            layers.Dense(hidden_dim * 2, activation='relu'), 
            layers.Dropout(0.1),  # Dropout for regularization
            layers.Dense(hidden_dim)
        ])
        self.output_layer = layers.Dense(class_count)

        # Compile the model
        optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0) # Clip gradients to prevent exploding gradients
        loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.05)
        self.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
        
        # Mixer layer to assign dynamic weights to each agent's logits
        self.mixer = layers.Dense(1)


    def call(self, agent_outputs, agents, p_hedge=None, training=False):
        """
        agent_outputs: list of dicts with keys: 
          - 'belief': numpy array or tensor of shape [class_count]
          - 'agent_id': int (0-based)
          - 'modality_id': int (0-based)
        Returns: 
          - fused_output: softmax tensor of shape [class_count]
        """
        x = []
        for out in agent_outputs:
            # Agent's prediction
            belief = tf.convert_to_tensor(out['belief'], dtype=tf.float32)
            # Embed identity
            agent_id = tf.convert_to_tensor(out['agent_id'], dtype=tf.int32)
            modality_id = tf.convert_to_tensor(out['modality_id'], dtype=tf.int32)
            hedge_weight = tf.convert_to_tensor(out["hedge_weight"], dtype=tf.float32)
            hedge_weight = tf.expand_dims(hedge_weight, axis=0) # shape [1]

            agent_vec = self.agent_embeddings(tf.expand_dims(agent_id, axis=0))
            modality_vec = self.modality_embeddings(tf.expand_dims(modality_id, axis=0))
            identity_vec = tf.concat([agent_vec, modality_vec], axis=-1) # shape [1, 2*embedding_dim]
            identity_vec = tf.reshape(identity_vec, [-1]) # shape [2*embedding_dim]
            if self.use_hedge_feat:
                combined = tf.concat([
                    belief,
                    identity_vec,
                    hedge_weight
                ], axis=0)
            else:
                combined = tf.concat([
                    belief,
                    identity_vec
                ], axis=0)
            x.append(combined)

        x = tf.stack(x, axis=0) # shape [N, K + 2E + 1]
        x = tf.expand_dims(x, axis=0)  # [1 (batch_num), N, K + 2E + 1]

        # Project to hidden_dim first
        x_proj = self.input_proj(x) # [1 (batch_num), N, H]

        # Attention Block (self-attention)
        attn_out = self.attn_block(query=x_proj, key=x_proj, value=x_proj, training=training) # [1, N, H]
        attn_out = self.attn_norm1(attn_out + x_proj)  # Residual connection + norm

        # Feed-forward network
        # Encode context-aware representations per agent (after attention)
        ffn_out = self.ffn(attn_out, training=training) # [1, N, H]
        ffn_out = self.attn_norm2(ffn_out + attn_out)  # Residual connection + norm
        ffn_out = tf.squeeze(ffn_out, axis=0)  # [N, H] removing batch dimension

        # Predict class logits for each agent independently -> o
        attn_logits = self.output_layer(ffn_out) # [N, K]
        # Assign dynamic weights to each agent's logits -> s
        agent_weights = self.mixer(ffn_out) # [N, 1]

        # agent_weights: [N, 1] -> π
        pi = tf.nn.softmax(agent_weights, axis=0)         # [N,1]
        pi = tf.squeeze(pi, axis=-1)                      # [N]

        # Store per-agent logits and π
        self.last_o = attn_logits                         # [N, K]
        self.last_pi = pi

        # Fused attention logits and probs
        aggregated_logits = tf.reduce_sum(tf.expand_dims(pi, -1) * attn_logits, axis=0)  # [K]
        r = tf.nn.softmax(aggregated_logits)              # [K]
        self.last_r = r

        # Hedge path: p_Hedge = Σ_i w_i * p_i
        if p_hedge is None:
            p_hedge = mix_beliefs_hedge(agent_outputs, agents, epsilon=self.epsilon)  # [K]
        else:
            p_hedge = tf.convert_to_tensor(p_hedge, dtype=tf.float32)
        self.last_p_hedge = p_hedge

        # Produce of Experts (PoE) fusion
        lam = self.lambda_poe
        log_p = (1.0 - lam) * tf.math.log(p_hedge + self.epsilon) + lam * tf.math.log(r + self.epsilon)  # [K]
        # Convert back to probs
        fused = tf.nn.softmax(log_p)  # [K]  (equivalent to renormalized exp(log_p))
        self.last_fused = fused
        return fused
    
    def train_on_single_example(self, agent_outputs, agents, true_label):
        with tf.GradientTape() as tape:
            fused = self.call(agent_outputs, agents, training=True)  # p_lambda (softmax(log_p))
            loss = tf.keras.losses.sparse_categorical_crossentropy(
                tf.convert_to_tensor([true_label], tf.int32),
                fused[tf.newaxis, ...], from_logits=False
            )
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return fused.numpy(), float(tf.reduce_mean(loss).numpy())