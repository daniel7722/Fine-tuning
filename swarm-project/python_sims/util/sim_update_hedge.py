import numpy as np

_state = {"correct_count": 0}

def update_hedge(hedge_cfg, fusion_unit, emissions, gt, agents, round_idx, csv_writer): 
    ETA = hedge_cfg.get("eta", 0.05)
    MIN_W = hedge_cfg.get("min_weight", 0.001)
    TEMP = hedge_cfg.get("temperature", 1.0)
    LOSS_TYPE = hedge_cfg.get("loss", "cross_entropy")
    if LOSS_TYPE == "cross_entropy":
        LOSS_TYPE = "nll"

    # Aggregate outputs from all agents
    fused = fusion_unit.call(agent_outputs=emissions)
    loss = fusion_unit.train_on_single_example(emissions, true_label=gt)

    # Update hedge weights based on losses
    new_weights = {}
    total_weight = 0.0
    for out in emissions:
        agent_id = out['agent_id']
        agent_i = next(a for a in agents if a.agent_id == agent_id)
        belief = np.array(out['belief'])
        if TEMP != 1.0:
            belief = np.power(belief, 1.0 / TEMP)
            belief = belief / np.sum(belief)
        if LOSS_TYPE == "nll":
            agent_loss = -np.log(belief[gt] + 1e-12)
        elif LOSS_TYPE == "01":
            pred = np.argmax(belief)
            agent_loss = 0.0 if pred == gt else 1.0
        else:
            agent_loss = 0.0
        updated_weight = agent_i.hedge_weight.numpy() * np.exp(-ETA * agent_loss)
        new_weights[agent_id] = updated_weight
        total_weight += updated_weight

    for agent_id, updated_weight in new_weights.items():
        normalised_weight = max(updated_weight, MIN_W)
        agent_i = next(a for a in agents if a.agent_id == agent_id)
        agent_i.hedge_weight.assign(normalised_weight)
    # Renormalize weights to sum to 1
    weight_sum = sum(a.hedge_weight.numpy() for a in agents)
    for a in agents:
        a.hedge_weight.assign(a.hedge_weight.numpy() / weight_sum)

    _state["correct_count"] += int(np.argmax(fused) == gt)
    correct_percentage = _state["correct_count"] / (round_idx + 1) * 100
    weights_by_id = {a.agent_id: float(a.hedge_weight.numpy()) for a in agents}
    hedge_weights = [weights_by_id.get(i, np.nan) for i in sorted(weights_by_id)]
    csv_writer.writerow([
        round_idx,
        gt,
        f"{correct_percentage:.2f}%",
        f"{loss:.4f}",
        np.argmax(fused),
        *hedge_weights
    ])
    if round_idx % 100 == 0:
        print(
            f"""Round {round_idx}: 
                Loss: {loss:.4f}, 
                Correct: {correct_percentage:.2f}%, 
                Hedge Weights: {hedge_weights}"""
        )