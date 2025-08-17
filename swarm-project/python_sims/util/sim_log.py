import csv
import numpy as np
import yaml
from pathlib import Path

# NOTE: keep the same dated log dir used elsewhere to avoid breaking paths
with open("configs/sim_config.yaml") as f:
    sim_config = yaml.safe_load(f)
DATE = sim_config.get("date")
_LOG_DIR = Path(f"logs/{DATE}")

def _ensure_dir():
    _LOG_DIR.mkdir(parents=True, exist_ok=True)
    return _LOG_DIR

def setup_log_file(filename):    
    """Create the main round-by-round log CSV.
    Columns now include fused correctness and per-agent correctness flags,
    which makes it easy to compute per-class accuracy later via a groupby.
    """
    log_path = _ensure_dir()
    log_file = open(log_path / f"AVE_{filename}.csv", "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(
        [
            "round",
            "ground_truth",
            "agent0_pred", "agent1_pred",
            "hedge0", "hedge1",
            "fused_B0_pred", "acc_B0",
            "fused_M1_pred", "acc_M1",
            "cos_sim_beliefs",
        ]
    )
    return log_file, csv_writer

def write_log(csv_writer, round_idx, gt, preds, hedges, fused_b0_pred, acc_b0, fused_m1_pred, acc_m1, cos_sim):
    # --- Write CSV row (list-based, matches header in sim_log.py) ---
    csv_writer.writerow([
        round_idx,
        gt,
        preds[0], preds[1],
        hedges[0], hedges[1],
        fused_b0_pred, acc_b0,
        fused_m1_pred, acc_m1,
        cos_sim,
    ])

def write_log_entry(csv_writer, round_idx, gt, correct_percentage, loss, fused, hedge_weights, agent_predictions, agent_correct_flags):
    fused_pred = int(np.argmax(fused))
    fused_correct = int(fused_pred == gt)

    # Expect exactly 2 agents for now (agent 0 and 1). If more agents are added later,
    # consider switching to a tidy/melted format.
    a0_hw = hedge_weights[0] if len(hedge_weights) > 0 else np.nan
    a1_hw = hedge_weights[1] if len(hedge_weights) > 1 else np.nan
    a0_pred = agent_predictions[0] if len(agent_predictions) > 0 else np.nan
    a1_pred = agent_predictions[1] if len(agent_predictions) > 1 else np.nan
    a0_corr = int(agent_correct_flags[0]) if len(agent_correct_flags) > 0 else 0
    a1_corr = int(agent_correct_flags[1]) if len(agent_correct_flags) > 1 else 0

    csv_writer.writerow([
        round_idx,
        gt,
        fused_pred,
        fused_correct,
        f"{loss:.4f}",
        f"{correct_percentage:.2f}%",
        a0_hw,
        a1_hw,
        a0_pred,
        a0_corr,
        a1_pred,
        a1_corr,
    ])
    if round_idx % 100 == 0:
        print(
            f"""Round {round_idx}: 
                Loss: {loss:.4f}, 
                Correct: {correct_percentage:.2f}%, 
                Hedge Weights: {hedge_weights}"""
        )


def setup_perclass_file(filename_suffix="perclass"):
    """Create a sidecar CSV for tidy per-class accuracy snapshots.

    Tidy schema (easy for pivot/heatmap later):
    round, modality, class_id, acc, support
    """
    log_path = _ensure_dir()
    pc_file = open(log_path / f"AVE_{filename_suffix}.csv", "w", newline="", encoding="utf-8")
    writer = csv.writer(pc_file)
    writer.writerow(["round", "modality", "class_id", "acc", "support"])  # tidy format
    return pc_file, writer


def write_perclass_snapshot(perclass_writer, round_idx, modality_name, confusion_matrix):
    """Append one tidy block of per-class accuracy rows for a modality.

    confusion_matrix: np.ndarray shape (C, C) with rows = true, cols = pred
    """
    cm = np.asarray(confusion_matrix)
    correct = np.diag(cm)
    support = cm.sum(axis=1)
    with np.errstate(divide='ignore', invalid='ignore'):
        acc = np.divide(correct, support, out=np.zeros_like(correct, dtype=float), where=support > 0)
    for class_id, (a, s) in enumerate(zip(acc, support)):
        perclass_writer.writerow([round_idx, modality_name, class_id, float(a), int(s)])
    try:
        perclass_writer.writerow([]) 
    except Exception:
        pass