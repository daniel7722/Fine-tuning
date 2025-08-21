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
from util.sim_metrics import ConfusionTracker, per_class_metrics_from_cm, macro_micro_from_cm, top_confusions, write_per_class_csv


def eval_fusion_split(
    name,
    dataset,
    agents,
    fusion_unit,
    DATE,
    use_baseline=True,
    use_attention=True,
    max_batches=None,
    log_filename=None,
):
    """
    Evaluates agents, B0 (hedge mixture), and M1 (PoE) on a dataset.
    IMPORTANT: does NOT update hedge or train attention (pure eval).
    If log_filename is provided, writes a per-example CSV matching the training schema.
    """

    EPS = 1e-12
    K = getattr(fusion_unit, "class_count", None)
    cm_a0 = None
    cm_a1 = None
    cm_b0 = None
    cm_m1 = None

    # Optional CSV logging
    csv_writer = None
    log_file = None
    if log_filename is not None:
        log_file, csv_writer = setup_log_file(log_filename)

    n = 0
    correct_a0 = correct_a1 = 0
    correct_b0 = correct_m1 = 0

    for i, data in enumerate(dataset):
        if max_batches is not None and i >= max_batches:
            break

        # Ground truth
        gt = int(data["label"].numpy()) if hasattr(data["label"], "numpy") else int(data["label"])

        # Emissions (no training here)
        emissions = [agent.emit(data) for agent in agents]

        # Per-agent beliefs/preds/hedges
        beliefs = [np.array(e["belief"], dtype=np.float32) for e in emissions]  # [N][K]
        preds   = [int(b.argmax()) for b in beliefs]
        hedges  = [float(a.hedge_weight.numpy()) for a in agents]

        # Lazy init K from first beliefs if needed
        if K is None:
            K = int(beliefs[0].shape[0])

        # Init CMs once we know K
        if cm_a0 is None:
            cm_a0 = np.zeros((K, K), dtype=np.int64)
            cm_a1 = np.zeros((K, K), dtype=np.int64)
            if use_baseline:
                cm_b0 = np.zeros((K, K), dtype=np.int64)
            if use_attention:
                cm_m1 = np.zeros((K, K), dtype=np.int64)

        # Update agent CMs
        cm_a0[gt, preds[0]] += 1
        cm_a1[gt, preds[1]] += 1

        # Cosine agreement
        na = np.linalg.norm(beliefs[0]); nb = np.linalg.norm(beliefs[1])
        cos_sim = float(np.dot(beliefs[0], beliefs[1]) / (na * nb + 1e-12))

        # Agents correct
        correct_a0 += int(preds[0] == gt)
        correct_a1 += int(preds[1] == gt)

        # Baseline B0
        fused_b0_pred = ""
        acc_b0 = ""
        p_b0_gt = ""
        loss_b0 = ""
        if use_baseline:
            p_hedge = mix_beliefs_hedge(emissions, agents)   # [K]
            fused_b0_pred = int(p_hedge.argmax())
            acc_b0 = int(fused_b0_pred == gt)
            p_b0_gt = float(p_hedge[gt])
            loss_b0 = float(-np.log(max(p_b0_gt, EPS)))
            correct_b0 += acc_b0
            cm_b0[gt, fused_b0_pred] += 1

        # M1 (PoE)
        fused_m1_pred = ""
        acc_m1 = ""
        p_m1_gt = ""
        loss_m1 = ""
        pi0 = ""; pi1 = ""
        m1_conf_top1 = ""; m1_margin = ""
        if use_attention:
            probs = fusion_unit.call(emissions, agents, training=False).numpy()  # [K]
            fused_m1_pred = int(probs.argmax())
            acc_m1 = int(fused_m1_pred == gt)
            correct_m1 += acc_m1

            p_m1_gt = float(probs[gt])
            loss_m1 = float(-np.log(max(p_m1_gt, EPS)))
            # attention weights (if exposed)
            if hasattr(fusion_unit, "last_pi") and fusion_unit.last_pi is not None:
                _pi = np.asarray(fusion_unit.last_pi.numpy()).reshape(-1)
                if _pi.size >= 2:
                    pi0, pi1 = float(_pi[0]), float(_pi[1])
            # confidence + margin
            if probs.size >= 2:
                top2 = np.partition(probs, -2)[-2:]
                m1_conf_top1 = float(probs.max())
                m1_margin = float(top2.max() - top2.min())
            else:
                m1_conf_top1 = float(probs.max())
                m1_margin = 0.0
            cm_m1[gt, fused_m1_pred] += 1

        # Optional CSV row
        if csv_writer is not None:
            write_log(
                csv_writer, i, gt, preds, hedges,
                fused_b0_pred, acc_b0,
                fused_m1_pred, acc_m1,
                cos_sim,
                p_b0_gt, p_m1_gt,
                loss_b0, loss_m1,
                pi0, pi1,
                m1_conf_top1, m1_margin
            )

        n += 1

    def _summarize(name_short, cm, log_filename):
        if cm is None:
            return
        m = per_class_metrics_from_cm(cm)
        macro, micro = macro_micro_from_cm(cm)
        print(f"[EVAL:{name}::{name_short}] micro-acc={micro['f1']:.3f}  macro-F1={macro['f1']:.3f}")
        print(f"[EVAL:{name}::{name_short}] top confusions:", top_confusions(cm, k=5))
        # Write per-class CSV next to the per-example CSV (if any)
        if log_filename is not None:
            out_path = f"logs/{DATE}/{log_filename}_perclass_{name_short}.csv"
            write_per_class_csv(out_path, name_short, m)

    _summarize("agent0", cm_a0, log_filename)
    _summarize("agent1", cm_a1, log_filename)
    if use_baseline:
        _summarize("B0", cm_b0, log_filename)
    if use_attention:
        _summarize("M1", cm_m1, log_filename)

    if use_baseline and use_attention and (cm_b0 is not None) and (cm_m1 is not None):
        mb0 = per_class_metrics_from_cm(cm_b0)
        mm1 = per_class_metrics_from_cm(cm_m1)
        delta_acc = mm1["accuracy"] - mb0["accuracy"]
        delta_f1  = mm1["f1"]       - mb0["f1"]
        # print top uplift classes
        order = np.argsort(delta_acc)[::-1]
        top5 = [(int(k), float(delta_acc[k]), float(delta_f1[k])) for k in order[:5]]
        print(f"[EVAL:{name}] Top-5 per-class gains (acc, f1):", top5)
        if log_filename is not None:
            import csv
            out_path = f"logs/{DATE}/{log_filename}_perclass_DELTA_M1_minus_B0.csv"
            with open(out_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["class","delta_acc","delta_f1"])
                for k in range(K):
                    w.writerow([k, float(delta_acc[k]), float(delta_f1[k])])

    # Close log if opened
    if log_file is not None:
        log_file.close()

    # Summary
    n = max(n, 1)
    res = {
        "split": name,
        "n": n,
        "acc_agent0": correct_a0 / n,
        "acc_agent1": correct_a1 / n,
        "acc_b0": (correct_b0 / n) if use_baseline else None,
        "acc_m1": (correct_m1 / n) if use_attention else None,
    }
    print(f"[EVAL:{name}] n={n} | "
          f"a0={res['acc_agent0']:.3f}  a1={res['acc_agent1']:.3f}  "
          f"b0={('%.3f'%res['acc_b0']) if res['acc_b0'] is not None else '—'}  "
          f"m1={('%.3f'%res['acc_m1']) if res['acc_m1'] is not None else '—'}")
    return res


def main(filename: str, pretrain: bool):
    # Show available GPUs
    print(tf.config.list_physical_devices("GPU"))

    # Load configs
    with open("configs/sim_config.yaml") as f:
        sim_config = yaml.safe_load(f)
        
    DATE = sim_config.get("date")
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

    # Confusion tracker
    tracker = ConfusionTracker(
        num_classes=NUM_CLASSES,
        agent_ids=[agent.agent_id for agent in agents]
    )

    # --- Iterative (single-threaded) rounds
    # IMPORTANT: iterate train_data exactly once, 1 sample per round.
    # All agents see the same 'data' dict each round.
    emissions = []

    for round_idx, data in enumerate(train_fuse_dataset.take(MAX_ROUNDS)):
        # Pull ground-truth as python int
        gt = int(data["label"].numpy()) if hasattr(data["label"], "numpy") else int(data["label"])

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
        
        VAL_EVERY = sim_config.get("val_every", 500)  # add to your yaml if you want
        if (round_idx + 1) % VAL_EVERY == 0:
            # Mid-training snapshot (small subset):
            eval_fusion_split(
                name=f"val@{round_idx+1}",
                dataset=val_dataset.take(200),
                agents=agents,
                fusion_unit=fusion_unit,
                DATE=DATE,
                use_baseline=USE_BASELINE,
                use_attention=USE_ATTENTION,
                log_filename=f"VAL_eval_round{round_idx+1}"   # will write logs/<DATE>
            )

        losses = compute_agent_loss(emissions, gt)
        _ = update_hedge(agents, losses)
        agent_preds_by_id = {e["agent_id"]: e["prediction"] for e in emissions}
        tracker.update(gt, fused_m1_pred if USE_ATTENTION else fused_b0_pred, agent_preds_by_id)

    
    # Final full validation:
    eval_fusion_split(
        name="val_final",
        dataset=val_dataset,
        agents=agents,
        fusion_unit=fusion_unit,
         DATE=DATE,
        use_baseline=USE_BASELINE,
        use_attention=USE_ATTENTION,
        log_filename="VAL_final_eval"
    )

    # Final test (this is the one you’ll analyze for the paper):
    eval_fusion_split(
        name="test_final",
        dataset=test_dataset,
        agents=agents,
        fusion_unit=fusion_unit,
         DATE=DATE,
        use_baseline=USE_BASELINE,
        use_attention=USE_ATTENTION,
        log_filename="TEST_final_eval"
    )

    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", required=True, help="Name for log file for this run")
    parser.add_argument("-p", "--pretrain", action="store_true", help="Whether to pretrain agents")
    args = parser.parse_args()

    run_name = args.name.strip()
    main(run_name, pretrain=args.pretrain)