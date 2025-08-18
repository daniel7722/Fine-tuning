import numpy as np, pandas as pd
from math import log

df = pd.read_csv("./logs/2025-08-18/AVE_additional_metrics.csv")

# 1) Final accuracies on test rows only (if you mark them), otherwise overall:
accs = {
    "Agent0": (df["agent0_pred"]==df["ground_truth"]).mean(),
    "Agent1": (df["agent1_pred"]==df["ground_truth"]).mean(),
    "B0": df["acc_B0"].mean(),
    "M1": df["acc_M1"].mean()
}
print(accs)

# 2) McNemar on M1 vs B0
# b = count(M1 correct, B0 wrong), c = count(M1 wrong, B0 correct)
m1c = df["acc_M1"]==1
b0c = df["acc_B0"]==1
b = int(( m1c & ~b0c ).sum())
c = int((~m1c &  b0c ).sum())
from statsmodels.stats.contingency_tables import mcnemar
print("McNemar:", mcnemar([[0,c],[b,0]], exact=True if b+c<25 else False))

# 3) Cosine buckets: show where fusion helps
bins = np.quantile(df["cos_sim_beliefs"], [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1])
df["cos_bin"] = pd.cut(df["cos_sim_beliefs"], bins, include_lowest=True)
grp = df.groupby("cos_bin")[["acc_B0","acc_M1"]].mean()
print(grp)

# 4) NLL and regret (requires p_b0_gt and p_m1_gt)
EPS=1e-12
if "p_b0_gt" in df and "p_m1_gt" in df:
    df["_nll_b0"] = -np.log(df["p_b0_gt"].clip(EPS))
    df["_nll_m1"] = -np.log(df["p_m1_gt"].clip(EPS))
    df["_cum_b0"] = df["_nll_b0"].cumsum()
    df["_cum_m1"] = df["_nll_m1"].cumsum()
    # baseline to best single agent by gt-prob (optional if you add p0_gt/p1_gt)
    # regret example: df["_cum_m1"] - min(df["_cum_agent0"], df["_cum_agent1"])
    print("Final cum NLL:", df[["_cum_b0","_cum_m1"]].iloc[-1].to_dict())

# 5) Calibration (ECE) for M1 using top-1 confidence
def ece(probs, correct, n_bins=10):
    bins = np.linspace(0,1,n_bins+1)
    idx = np.digitize(probs, bins) - 1
    ece=0.0
    for b in range(n_bins):
        mask = idx==b
        if mask.any():
            acc = correct[mask].mean()
            conf= probs[mask].mean()
            ece += (mask.mean()) * abs(acc - conf)
    return ece

if "m1_conf_top1" in df:
    e = ece(df["m1_conf_top1"].values, df["acc_M1"].values.astype(bool), n_bins=10)
    print("ECE(M1):", e)