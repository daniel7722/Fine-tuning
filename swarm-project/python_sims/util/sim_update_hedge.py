import numpy as np


def compute_agent_loss(emissions, gt): 
    """Returns list[float] NLL losses aligned with emissions order."""
    losses = []
    for out in emissions: 
        p = np.asarray(out['belief'], dtype=np.float64)
        losses.append(-np.log(p[gt] + 1e-12))
    return losses

def update_hedge(agents, losses, eta=0.05, min_weight=1e-3): 
    """In-place Hedge update on each agent. Returns list[float] new normalised weights."""
    # multiply-update + floor
    for a, loss in zip(agents, losses): 
        w = float(a.hedge_weight.numpy()) * np.exp(-eta * loss)
        a.hedge_weight.assign(max(w, min_weight))

    # renormalise
    s = sum(float(a.hedge_weight.numpy()) for a in agents)
    s = s if s > 0 else 1.0
    for a in agents: 
        a.hedge_weight.assign(float(a.hedge_weight.numpy()) / s)
    return [float(a.hedge_weight.numpy()) for a in agents]

def mix_beliefs_hedge(emissions, agents, epsilon=1e-12): 
    """p_hedge = Σ_i w_i * p_i"""
    weights = np.array([float(a.hedge_weight.numpy()) for a in agents], dtype=np.float64)
    P = np.stack([np.asarray(e["belief"], dtype=np.float64) for e in emissions], axis=0)  # [N, K]
    p = (weights[:, None] * P).sum(axis=0)  # [K]
    s = p.sum()
    return (p / (s + epsilon)).astype(np.float32)