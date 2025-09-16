import time
import argparse
import numpy as np
import tensorflow as tf
import yaml

from agent import VisionAgent, AudioAgent
from fusion_unit import FusionUnit
from util.sim_load_data import load_data
from util.sim_pretrain_agent import pre_train_agents
from util.sim_log import setup_log_file, write_log
from util.sim_update_hedge import compute_agent_loss, update_hedge, mix_beliefs_hedge
from util.sim_metrics import ConfusionTracker
from util.sim_eval import eval_fusion_split

def main(pretrain: bool):
    # Show available GPUs
    print(tf.config.list_physical_devices("GPU"))

    # Load configs
    with open("configs/sim_config.yaml", encoding="utf-8") as f:
        sim_config = yaml.safe_load(f)
        
    DATE = sim_config.get("date")
    NUM_CLASSES   = sim_config.get("num_classes", 2)
    NUM_AGENTS    = sim_config.get("num_agents", 2)      # you only instantiate 2 below
    NUM_MODALITIES= sim_config.get("num_modalities", 2)
    MAX_ROUNDS    = sim_config.get("max_rounds", 5000)

    USE_BASELINE = sim_config.get("use_baseline", True)
    USE_ATTENTION = sim_config.get("use_attention", False)
    ALLOW_HEDGE_UPDATE = sim_config.get("allow_hedge_update", False)
    ETA = sim_config.get("eta", 0.05)
    LAMBDA_POE = sim_config.get("lambda_poe", 0.5)  # PoE lambda
    USE_HEDGE_FEAT = sim_config.get("use_hedge_feat", False)  # whether to use Hedge weights as a feature in M1
    CORRUPTION = sim_config.get("corruption", {})
    corruption_enabled = CORRUPTION.get("corrupt", False)
    corrupt_agent = CORRUPTION.get("corrupt_agent", 1) 
    corrupt_start = CORRUPTION.get("corrupt_start", 500)
    corrupt_len = CORRUPTION.get("corrupt_len", 150)
    filename = sim_config.get("name", "default_run")

    # --- Fusion unit
    fusion_unit = FusionUnit(
        class_count=NUM_CLASSES,
        num_agents=NUM_AGENTS,
        num_modalities=NUM_MODALITIES,
        lambda_poe=LAMBDA_POE, 
        use_hedge_feat = USE_HEDGE_FEAT,
    )

    # --- Agents
    agents = [
        VisionAgent(agent_id=0, class_count=NUM_CLASSES),
        AudioAgent(agent_id=1, class_count=NUM_CLASSES),
    ]

    # --- Data
    train_pre_dataset, val_pre_dataset, train_fuse_dataset, val_dataset, test_dataset = load_data()

    # --- Pretrain (as before)
    pre_train_agents(agents, train_pre_dataset, val_pre_dataset, pretrain)

    # --- Ask to continue into online rounds
    ans = input("Continue training? (y/n): ").strip().lower()
    if ans != "y":
        print("Exiting without training.")
        return
    print("Starting training...")

    # --- Logging
    log_file, csv_writer = setup_log_file(filename)

    # --- Confusion tracker
    tracker = ConfusionTracker(
        num_classes=NUM_CLASSES,
        agent_ids=[agent.agent_id for agent in agents]
    )

    emissions = []

    for round_idx, data in enumerate(train_fuse_dataset.take(MAX_ROUNDS)):
        # Pull ground-truth as python int
        gt = int(data["label"].numpy()) if hasattr(data["label"], "numpy") else int(data["label"])

        if round_idx < 3:
            print(f"[Round {round_idx}] GT label = {gt}")

        emissions.clear()
        for agent in agents:
            out = agent.emit(data)
            if round_idx < 3:
                print(f"  Agent {agent.agent_id} pred={out['prediction']} "
                      f"correct={out['correct']} hedge={out['hedge_weight']:.4f}")
            emissions.append(out)

        # Gather beliefs, preds, hedges for baseline 
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
            #  Baseline B0: hedge mixture (weighted average) 
            p_hedge = mix_beliefs_hedge(emissions, agents)
            fused_b0_pred   = int(np.argmax(p_hedge))
            acc_b0          = int(fused_b0_pred == gt)

        fused_m1_pred = ""
        acc_m1 = ""
        if USE_ATTENTION:
            m1_probs, loss = fusion_unit.train_on_single_example(emissions, agents, gt)
            fused_m1_pred = int(np.argmax(m1_probs))
            acc_m1 = int(fused_m1_pred == gt)

        EPS = 1e-12

        # Defaults if a path isn’t used this round
        p_b0_gt = ""
        p_m1_gt = ""
        loss_b0 = ""
        loss_m1 = ""
        pi0 = ""
        pi1 = ""
        m1_conf_top1 = ""
        m1_margin = ""

        # --- Baseline path (B0) ---
        if USE_BASELINE:
            # probability assigned to the ground-truth by Hedge mixture
            p_b0_gt = float(p_hedge[gt])
            loss_b0 = float(-np.log(max(p_b0_gt, EPS)))

        # --- Attention path (M1) ---
        if USE_ATTENTION:
            probs = np.asarray(m1_probs, dtype=np.float64)  # M1 PoE probs, shape [K]
            p_m1_gt = float(probs[gt])
            loss_m1 = float(-np.log(max(p_m1_gt, EPS)))

            # attention mixing weights π (optional taps)
            if hasattr(fusion_unit, "last_pi") and fusion_unit.last_pi is not None:
                _pi = np.asarray(fusion_unit.last_pi.numpy()).reshape(-1)
                if _pi.size >= 2:
                    pi0, pi1 = float(_pi[0]), float(_pi[1])

            # confidence + margin (top1 - top2)
            if probs.size >= 2:
                top2 = np.partition(probs, -2)[-2:]  # unsorted two largest
                m1_conf_top1 = float(probs.max())
                m1_margin = float(top2.max() - top2.min())
            else:
                m1_conf_top1 = float(probs.max())
                m1_margin = 0.0
        write_log(csv_writer, round_idx, gt, preds, hedges,
                  fused_b0_pred, acc_b0, fused_m1_pred, acc_m1,
                  cos_sim, p_b0_gt, p_m1_gt,
                  loss_b0, loss_m1, pi0, pi1,
                  m1_conf_top1, m1_margin)
       
        VAL_EVERY = sim_config.get("val_every", 500)
        if (round_idx + 1) % VAL_EVERY == 0:
            # Mid-training snapshot (small subset):
            eval_fusion_split(
                name=f"val@{round_idx+1}",
                dataset=val_dataset.take(200),
                agents=agents,
                fusion_unit=fusion_unit,
                DATE=DATE,
                eta=ETA,
                use_baseline=USE_BASELINE,
                use_attention=USE_ATTENTION,
                log_filename=f"VAL_eval_round{round_idx+1}",
            )

        losses = compute_agent_loss(emissions, gt)
        _ = update_hedge(agents, losses, eta=ETA)
        agent_preds_by_id = {e["agent_id"]: e["prediction"] for e in emissions}
        tracker.update(gt, fused_m1_pred if USE_ATTENTION else fused_b0_pred, agent_preds_by_id)

    
    # Final full validation:
    eval_fusion_split(
        name="val_final",
        dataset=val_dataset,
        agents=agents,
        fusion_unit=fusion_unit,
        DATE=DATE,
        eta=ETA,
        use_baseline=USE_BASELINE,
        use_attention=USE_ATTENTION,
        log_filename="VAL_final_eval",
    )

    # Final test:
    eval_fusion_split(
        name="test_final",
        dataset=test_dataset,
        agents=agents,
        fusion_unit=fusion_unit,
        DATE=DATE,
        eta=ETA,
        use_baseline=USE_BASELINE,
        use_attention=USE_ATTENTION,
        allow_hedge_update=ALLOW_HEDGE_UPDATE,
        log_filename="TEST_final_eval",
        corruption=corruption_enabled,
        corrupt_start=corrupt_start,
        corrupt_len=corrupt_len,
        corrupt_agent=corrupt_agent
    )

    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pretrain", action="store_true", help="Whether to pretrain agents")
    args = parser.parse_args()
    _t0 = time.time()
    main(pretrain=args.pretrain)
    _elapsed = time.time() - _t0
    _h = int(_elapsed // 3600)
    _m = int((_elapsed % 3600) // 60)
    _s = int(_elapsed % 60)
    print(f"[TIME] Total run time: {_h}h {_m}m {_s}s")