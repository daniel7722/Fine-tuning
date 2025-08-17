import argparse
import numpy as np
import tensorflow as tf
import yaml

from agent import VisionAgent, AudioAgent
from fusion_unit import FusionUnit
from util.sim_load_data import load_data
from util.sim_pretrain_agent import pre_train_agents
from util.sim_log import setup_log_file, write_log
from util.sim_update_hedge import *
from util.sim_metrics import ConfusionTracker


def main(filename: str, pretrain: bool):
    # Show available GPUs
    print(tf.config.list_physical_devices("GPU"))

    # Load configs
    with open("configs/sim_config.yaml") as f:
        sim_config = yaml.safe_load(f)

    NUM_CLASSES   = sim_config.get("num_classes", 2)
    NUM_AGENTS    = sim_config.get("num_agents", 2)      # you only instantiate 2 below
    NUM_MODALITIES= sim_config.get("num_modalities", 2)
    MAX_ROUNDS    = sim_config.get("max_rounds", 5000)

    USE_BASELINE = sim_config.get("use_baseline", True)
    USE_ATTENTION = sim_config.get("use_attention", False)
    LOG_ATTENTION = sim_config.get("log_attention", False)

    # --- Fusion unit
    fusion_unit = FusionUnit(
        class_count=NUM_CLASSES,
        num_agents=NUM_AGENTS,
        num_modalities=NUM_MODALITIES,
    )

    # --- Agents
    agents = [
        VisionAgent(agent_id=0, class_count=NUM_CLASSES),
        AudioAgent(agent_id=1, class_count=NUM_CLASSES),
    ]

    # --- Data
    train_data, val_data, test_data = load_data()

    # --- Pretrain (as before)
    pre_train_agents(agents, train_data, val_data, pretrain)

    # --- Ask to continue into online rounds
    ans = input("Continue training? (y/n): ").strip().lower()
    if ans != "y":
        print("Exiting without training.")
        return
    print("Starting training...")

    # --- Logging
    log_file, csv_writer = setup_log_file(filename)

    # Confusion tracker
    tracker = ConfusionTracker(
        num_classes=NUM_CLASSES,
        agent_ids=[agent.agent_id for agent in agents]
    )

    # --- Iterative (single-threaded) rounds
    # IMPORTANT: iterate train_data exactly once, 1 sample per round.
    # All agents see the same 'data' dict each round.
    emissions = []

    for round_idx, data in enumerate(train_data.take(MAX_ROUNDS)):
        # Pull ground-truth as python int
        gt_t = int(data["label"].numpy())
        gt = int(gt_t.numpy()) if isinstance(gt_t, tf.Tensor) else int(gt_t)

        # Optional sanity print for the first few rounds only
        if round_idx < 3:
            print(f"[Round {round_idx}] GT label = {gt}")

        emissions.clear()
        for agent in agents:
            out = agent.emit(data)
            # Optionally sanity-print what the agent believes on the first few rounds
            if round_idx < 3:
                print(f"  Agent {agent.agent_id} pred={out['prediction']} "
                      f"correct={out['correct']} hedge={out['hedge_weight']:.4f}")
            emissions.append(out)

        # --- Gather beliefs, preds, hedges for baseline ---
        beliefs = [np.array(e["belief"], dtype=np.float32) for e in emissions]   # [B_v, B_a]
        preds   = [int(np.argmax(b)) for b in beliefs]
        hedges  = [float(a.hedge_weight.numpy()) for a in agents]                # [h_v, h_a]

        # Cosine similarity between beliefs
        def _cos(a, b):
            na = np.linalg.norm(a); nb = np.linalg.norm(b)
            return float(np.dot(a, b) / (na * nb + 1e-12))
        cos_sim = _cos(beliefs[0], beliefs[1])

        acc_b0 = ""
        fused_b0_pred = ""
        if USE_BASELINE:
            # --- Baseline B0: hedge mixture (weighted average) ---
            p_hedge = mix_beliefs_hedge(emissions, agents)
            fused_b0_pred   = int(np.argmax(p_hedge))
            acc_b0          = int(fused_b0_pred == gt)

        # --- Optionally compute M1 fused prediction ---
        fused_m1_pred = ""
        acc_m1 = ""
        if USE_ATTENTION:
            m1_probs, loss = fusion_unit.train_on_single_example(emissions, agents, gt)
            fused_m1_pred = int(np.argmax(m1_probs))
            acc_m1 = int(fused_m1_pred == gt)

        write_log(csv_writer, round_idx, gt, preds, hedges,
                  fused_b0_pred, acc_b0, fused_m1_pred, acc_m1,
                  cos_sim)

        losses = compute_agent_loss(emissions, gt)
        _ = update_hedge(agents, losses)
        agent_preds_by_id = {e["agent_id"]: e["prediction"] for e in emissions}
        tracker.update(gt, fused_m1_pred if USE_ATTENTION else fused_b0_pred, agent_preds_by_id)

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name for log file for this run")
    parser.add_argument("-p", "--pretrain", action="store_true", help="Whether to pretrain agents")
    args = parser.parse_args()

    run_name = args.name.strip()
    main(run_name, pretrain=args.pretrain)