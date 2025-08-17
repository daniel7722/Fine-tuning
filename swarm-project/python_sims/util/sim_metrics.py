# util/sim_metrics.py
import numpy as np

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

# Optional: tidy per-class snapshot writer hooks can live here if you still want them later.