import numpy as np
import pytest
import tensorflow as tf
from agent_interface import Agent

class DummyModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(10, activation='relu', use_bias=False)
        # Build the dense layer so weights can be created
        self.dense.build((None, 1))
        # Initialize weights to ones
        self.dense.set_weights([np.ones((1, 10))])

    def call(self, inputs, training=False):
        return self.dense(inputs)

@pytest.fixture
def config(): 
    return {
      'optimizer': 'adam',
      'learning_rate': 0.001,
      'batch_per_round': 1,
      'loss_fn': tf.keras.losses.MeanSquaredError(),
      'log_interval': 1
    }

@pytest.fixture
def data_loader(): 
    def gen(): 
        while True: 
            yield np.array([[1.0]]), np.array([[2.0]])
    return gen()

def test_get_set_weights_roundtrip(config, data_loader): 
    agent = Agent(0, data_loader, DummyModel, config)
    orig = agent._get_weights().copy()
    perturbed = orig + 5.0
    agent._set_weights(perturbed)
    new = agent._get_weights()
    assert np.allclose(new, perturbed)

def test_compute_delta_and_apply(config, data_loader):
    agent = Agent(0, data_loader, DummyModel, config)
    base = agent._get_weights().copy()
    # simulate a weight change
    agent._set_weights(base + 3.0)
    agent.compute_delta()
    assert np.allclose(agent.delta, 3.0)
    # apply two peer deltas of +1 and +5
    agent._peer_deltas = [
        np.full((1, 10), 1.0).flatten(), 
        np.full((1, 10), 5.0).flatten()
    ]
    base = agent._get_weights().copy()
    agent.apply_deltas([])
    expected =  base+ (1 + 5 + 3)/3
    assert np.allclose(agent._get_weights(), expected)

def test_gossip_and_receive(config, data_loader):
    agent_a = Agent(0, data_loader, DummyModel, config)
    agent_b = Agent(1, data_loader, DummyModel, config)
    agent_a.delta = 3.0
    agent_a.gossip([agent_b])
    deltas = agent_b._peer_deltas
    assert len(deltas) == 1
    assert np.allclose(deltas[0], agent_a.delta)

def test_train_changes_weights(config, data_loader):
    agent = Agent(0, data_loader, DummyModel, config)
    before = agent._get_weights().copy()
    agent.train(num_batches=1)
    after = agent._get_weights()
    assert not np.allclose(before, after), "Weights should change after training"

def test_reset_clears_state(config, data_loader):
    agent = Agent(0, data_loader, DummyModel, config)
    agent.train(num_batches=1)
    agent.compute_delta()
    agent.receive_delta(agent.delta)
    agent.reset()
    assert agent.delta is None
    assert agent._peer_deltas == []
    assert np.allclose(agent._prev_weights, agent._get_weights())