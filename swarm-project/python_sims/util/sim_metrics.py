# util/sim_metrics.py
import numpy as np
import csv

class ConfusionTracker:
    def __init__(self, num_classes, agent_ids):
        self.C = int(num_classes)
        self.cm_fusion = np.zeros((self.C, self.C), dtype=np.int64)
        self.cm_agents = {aid: np.zeros((self.C, self.C), dtype=np.int64) for aid in agent_ids}
        self.correct_count = 0

    def update(self, gt, fused_pred, agent_preds_by_id):
        self.cm_fusion[gt, fused_pred] += 1
        for aid, pred in agent_preds_by_id.items():
            self.cm_agents[aid][gt, pred] += 1
        self.correct_count += int(fused_pred == gt)

    def accuracy(self, rounds_seen):
        if rounds_seen <= 0:
            return 0.0
        return self.correct_count / float(rounds_seen)

def _safe_div(a, b, eps=1e-12):
    return np.where(b > 0, a / (b + eps), 0.0)

def per_class_metrics_from_cm(cm: np.ndarray):
    """
    Returns a dict with per-class precision, recall, f1, support, pred_support, accuracy.
    accuracy_k = TP_k / support_k (row-normalized accuracy for class k).
    """
    cm = np.asarray(cm, dtype=np.int64)
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)      # row sums (true)
    pred_support = cm.sum(axis=0).astype(np.float64) # col sums (pred)

    recall = _safe_div(tp, support)                 # sensitivity
    precision = _safe_div(tp, pred_support)         # positive predictive value
    f1 = _safe_div(2.0 * precision * recall, precision + recall)
    acc = _safe_div(tp, support)                    # same as recall here

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "pred_support": pred_support,
        "accuracy": acc,
    }

def macro_micro_from_cm(cm: np.ndarray):
    """Macro and micro P/R/F1 from a confusion matrix."""
    m = per_class_metrics_from_cm(cm)
    macro = {
        "precision": float(np.mean(m["precision"])),
        "recall":    float(np.mean(m["recall"])),
        "f1":        float(np.mean(m["f1"])),
    }
    # micro from CM
    tp = float(np.trace(cm))
    total = float(cm.sum())
    micro_acc = tp / (total + 1e-12)
    micro = {"precision": micro_acc, "recall": micro_acc, "f1": micro_acc}
    return macro, micro

def top_confusions(cm: np.ndarray, k=5):
    """
    Return list of (true, pred, count) for the k largest off-diagonal confusions.
    """
    cm = np.asarray(cm, dtype=np.int64)
    x = cm.copy()
    np.fill_diagonal(x, 0)
    idx = np.dstack(np.unravel_index(np.argsort(x.ravel())[::-1], x.shape))[0]
    out = []
    for (i, j) in idx:
        if x[i, j] <= 0:
            break
        out.append((int(i), int(j), int(x[i, j])))
        if len(out) >= k:
            break
    return out

def write_per_class_csv(path: str, model_name: str, metrics: dict):
    """
    Write per-class metrics to CSV: class,model,precision,recall,f1,accuracy,support,pred_support
    """
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class","model","precision","recall","f1","accuracy","support","pred_support"])
        K = len(metrics["precision"])
        for k in range(K):
            w.writerow([
                k, model_name,
                float(metrics["precision"][k]),
                float(metrics["recall"][k]),
                float(metrics["f1"][k]),
                float(metrics["accuracy"][k]),
                int(metrics["support"][k]),
                int(metrics["pred_support"][k]),
            ])