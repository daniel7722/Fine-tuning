import numpy as np
import csv

from util.sim_log import setup_log_file, write_log
from util.sim_update_hedge import mix_beliefs_hedge
from util.sim_metrics import per_class_metrics_from_cm, macro_micro_from_cm, top_confusions, write_per_class_csv
from util.sim_update_hedge import compute_agent_loss, update_hedge

def eval_fusion_split(
    name,
    dataset,
    agents,
    fusion_unit,
    DATE,
    eta=0.05,
    use_baseline=True,
    use_attention=True,
    allow_hedge_update=False,
    max_batches=None,
    log_filename=None,
    corruption=False,
    corrupt_start=500, 
    corrupt_len=150, 
    corrupt_agent=1
):
    """
    Evaluates agents, B0 (hedge mixture), and M1 (PoE) on a dataset.
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

    extras_writer = None
    extras_file = None
    if log_filename is not None:
        extras_path = f"logs/{DATE}/{log_filename}_corruption_signals.csv"
        extras_file = open(extras_path, "w", newline="", encoding="utf-8")
        extras_writer = csv.writer(extras_file)
        extras_writer.writerow([
            "idx",
            "is_corrupt",
            "b0_conf",
            "ent_agent0",
            "ent_agent1",
            "log_odds_audio_vs_vision",
            "loss_agent0",
            "loss_agent1",
        ])

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
        is_corrupt = bool(corruption and (i >= corrupt_start) and (i < corrupt_start + corrupt_len))
        if is_corrupt:
            b = np.asarray(emissions[corrupt_agent]["belief"], dtype=np.float32)
            K = b.shape[-1]
            b = np.full(K, 1.0 / K, dtype=np.float32)  # uniform belief
            emissions[corrupt_agent]["belief"] = b

        # Per-agent beliefs/preds/hedges
        beliefs = [np.array(e["belief"], dtype=np.float32) for e in emissions]  # [N][K]
        preds   = [int(b.argmax()) for b in beliefs]
        hedges  = [float(a.hedge_weight.numpy()) for a in agents]

        # Per-agent entropies (natural log base)
        entropies = [-float(np.sum(b * np.log(b + EPS))) for b in beliefs]
        # Hedge log-odds (audio vs vision). Assumes agent0=vision, agent1=audio.
        log_odds = float(np.log((hedges[1] + EPS) / (hedges[0] + EPS)))

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
        b0_conf = ""
        if use_baseline:
            p_hedge = mix_beliefs_hedge(emissions, agents)   # [K]
            fused_b0_pred = int(p_hedge.argmax())
            acc_b0 = int(fused_b0_pred == gt)
            p_b0_gt = float(p_hedge[gt])
            loss_b0 = float(-np.log(max(p_b0_gt, EPS)))
            b0_conf = float(p_hedge.max())
            correct_b0 += acc_b0
            cm_b0[gt, fused_b0_pred] += 1
        else:
            fused_b0_pred = ""
            acc_b0 = ""
            p_b0_gt = ""
            loss_b0 = ""

        # M1 (PoE)
        fused_m1_pred = ""
        acc_m1 = ""
        p_m1_gt = ""
        loss_m1 = ""
        pi0 = ""; pi1 = ""
        m1_conf_top1 = ""
        m1_margin = ""
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

        # Compute per-agent losses for logging (always), before any optional hedge update
        agent_losses = compute_agent_loss(emissions, gt)

        if extras_writer is not None:
            # agent0 assumed vision, agent1 audio for naming consistency
            loss_a0 = float(agent_losses[0]) if isinstance(agent_losses, (list, tuple, np.ndarray)) else float(agent_losses.get(0, 0.0))
            loss_a1 = float(agent_losses[1]) if isinstance(agent_losses, (list, tuple, np.ndarray)) else float(agent_losses.get(1, 0.0))
            extras_writer.writerow([
                i,
                int(is_corrupt),
                ("" if b0_conf == "" else float(b0_conf)),
                float(entropies[0]),
                float(entropies[1]),
                float(log_odds),
                loss_a0,
                loss_a1,
            ])
        
        if allow_hedge_update:
            # Update Hedge weights in-place
            losses = compute_agent_loss(emissions, gt)
            update_hedge(agents, losses, eta=eta)

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
            out_path = f"logs/{DATE}/{log_filename}_perclass_DELTA_M1_minus_B0.csv"
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(["class","delta_acc","delta_f1"])
                for k in range(K):
                    w.writerow([k, float(delta_acc[k]), float(delta_f1[k])])

    # Close logs if opened
    if log_file is not None:
        log_file.close()
    if extras_file is not None:
        extras_file.close()

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