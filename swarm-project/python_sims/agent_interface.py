import threading
import numpy as np
import tensorflow as tf


class Agent:
    """
    Agent interface for swarm learning simulation.
    Each agent handles its own local training, delta computation, gossiping, aggregation, and evaluation.
    """

    def __init__(self, agent_id, data_loader, model_factory, config):
        """
        :param agent_id: Unique identifier for this agent.
        :param data_loader: Callable or generator yielding (images, targets) batches.
        :param model_factory: Callable that returns a fresh model instance.
        :param config: Dictionary of hyperparameters and settings.
        """
        self.agent_id = agent_id
        self.data_loader = data_loader
        self.model = model_factory()
        self.config = config
        self.optimizer = self._init_optimizer()
        self.loss_fn = self._init_loss_fn()
        self._prev_weights = self._get_weights()
        self.delta = None
        self._peer_deltas = []  # Buffer for received deltas
        self._lock = threading.Lock()  # Thread-safe access to deltas

        # PSO state
        self.pbest_weights = self._prev_weights.copy()
        self.pbest_loss = float("inf")
        self.velocity = np.zeros_like(self._prev_weights)
        # PSO parameters
        self.c1 = self.config.get("pso_c1", 1.5)  # Cognitive coefficient
        self.c2 = self.config.get("pso_c2", 1.5)  # Social coefficient
        self.w = self.config.get("pso_w", 0.9)  # Inertia weight for velocity update

    def _init_optimizer(self):
        """
        Initialize optimizer based on config (e.g., SGD, Adam).
        """
        optimizer_type = self.config.get("optimizer", "adam")
        learning_rate = self.config.get("learning_rate", 0.001)

        if optimizer_type == "adam":
            return tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer_type == "sgd":
            return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    def _init_loss_fn(self):
        """
        Initialize loss function based on config (e.g., categorical crossentropy).
        """
        loss_type = self.config.get("loss_fn", "categorical_crossentropy")

        if loss_type == "categorical_crossentropy":
            return tf.keras.losses.CategoricalCrossentropy(from_logits=True)
        elif loss_type == "sparse_categorical_crossentropy":
            return tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            raise ValueError(f"Unsupported loss function: {loss_type}")

    def _get_weights(self):
        """
        Extract current model weights as a flat list or numpy array.
        """
        return np.concatenate([w.flatten() for w in self.model.get_weights()])

    def _set_weights(self, weights):
        """
        Set model weights from a flat list or numpy array.
        :param weights: Numpy array of weights to set.
        """
        shapes = [w.shape for w in self.model.get_weights()]
        start = 0
        new_weights = []
        for shape in shapes:
            size = np.prod(shape)
            new_weights.append(weights[start : start + size].reshape(shape))
            start += size
        self.model.set_weights(new_weights)

    def train(self, num_batches=None):
        """
        Perform local training for a given number of batches.
        Updates the model in-place.
        """
        print(f"Agent {self.agent_id} starting local training")
        # Determine how many batches to train; default from config
        num = num_batches or self.config.get("batches_per_round", 2)
        loss = None
        loss_fn = self.loss_fn
        for step in range(num):
            imgs, targets = next(self.data_loader)

            with tf.GradientTape() as tape:
                logits = self.model(imgs, training=True)
                loss = loss_fn(targets, logits)
            grads = tape.gradient(loss, self.model.trainable_weights)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

            if step % self.config.get("log_interval", 1) == 0:
                print(f"Agent {self.agent_id} - Loss: {loss.numpy()}")

        return loss

    def compute_delta(self):
        """
        Compute the weight delta since the last snapshot.
        Stores it in self.delta.
        """
        print(f"Agent {self.agent_id} computing delta")
        current = self._get_weights()
        # Delta = current - previous
        self.delta = current - self._prev_weights
        self._prev_weights = current.copy()

    def get_delta(self):
        """
        Return the most recently computed delta.
        """
        return self.delta

    def apply_deltas(self, deltas):
        """
        Aggregate a list of peer deltas with self.delta and apply to model.
        :param deltas: List of numpy arrays of the same shape as self.delta
        """
        print(
            f"Agent {self.agent_id} applying deltas from {len(self._peer_deltas)} peers"
        )
        assert all(d.shape == self.delta.shape for d in deltas)
        with self._lock:
            peer_deltas = list(self._peer_deltas)  # Copy to avoid
            self._peer_deltas.clear()  # Clear after aggregation
        all_deltas = peer_deltas + [self.delta]
        mean_delta = np.mean(all_deltas, axis=0)
        # Apply mean_delta to model weights
        final_weights = self._get_weights() + mean_delta
        self._set_weights(final_weights)

    def gossip(self, peer_agents):
        """
        Share self.delta with selected peers.
        :param peer_agents: List of Agent instances to send delta to.
        """
        print(f"Agent {self.agent_id} gossiping delta to {len(peer_agents)} peers")
        for peer in peer_agents:
            peer.receive_delta(self.delta)

    def receive_delta(self, delta):
        """
        Receive a delta from a peer; can buffer for aggregation.
        """
        # Buffer deltas externally, or implement as needed
        with self._lock:
            self._peer_deltas.append(delta)

    def evaluate(self, validation_loader):
        """
        Evaluate current model performance on a validation set.
        :param validation_loader: Iterable of (images, targets) for evaluation.
        :return: Metric dict (e.g., {'loss': value})
        """
        print(f"Agent {self.agent_id} evaluating model on validation set")
        # Run inference on validation_loader and compute metrics
        total_loss = 0.0
        num_samples = 0
        loss_fn = self.loss_fn
        for imgs, targets in validation_loader:
            logits = self.model(imgs, training=False)
            loss = loss_fn(targets, logits)
            total_loss += loss.numpy() * imgs.shape[0]
            num_samples += imgs.shape[0]
        avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

        print(f"Agent {self.agent_id} - Validation loss: {avg_loss}")
        if avg_loss < self.pbest_loss:
            self.pbest_loss = avg_loss
            self.pbest_weights = self._get_weights().copy()

        return {"loss": avg_loss}

    def reset(self):
        """
        Reset any internal state or buffers between simulation runs.
        """
        # Reset optimizer, weights snapshot, delta buffer, etc.
        self._prev_weights = self._get_weights()
        self.delta = None
        self._peer_deltas.clear()
        self.optimizer = self._init_optimizer()

    def apply_pso(self, global_best_weights):
        """
        PSO velocity+position update using personal best and global best.
        """
        current = self._get_weights()
        r1, r2 = np.random.rand(), np.random.rand()
        # Velocity update
        new_vel = (
            self.w * self.velocity
            + self.c1 * r1 * (self.pbest_weights - current)
            + self.c2 * r2 * (global_best_weights - current)
        )
        if self.config.get("pso_velocity_limit") is not None:
            new_vel = np.clip(
                new_vel,
                -self.config["pso_velocity_limit"],
                self.config["pso_velocity_limit"],
            )
        self.velocity = new_vel

        # Position update
        new_pos = current + new_vel
        self._set_weights(new_pos)
        self._prev_weights = new_pos.copy()
