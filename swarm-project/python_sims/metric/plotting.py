import os
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from math import log
import scipy.stats as stats
import seaborn as sns
import scienceplots

# --- Global style -----------------------------------------------------------
plt.style.use("science")
sns.set_theme(context="paper", style=None, rc=plt.rcParams)

# --- IO paths ---------------------------------------------------------------
with open("configs/sim_config.yaml", encoding="utf-8") as f:
    sim_config = yaml.safe_load(f)
DATE = sim_config.get("date")
name = sim_config.get("name")

TRAIN_CSV = f"./logs/{DATE}/AVE_{name}.csv"
TEST_CSV  = f"./logs/{DATE}/AVE_TEST_final_eval.csv"
AGENT0_PER_CLASS_CSV = f"./logs/{DATE}/TEST_final_eval_perclass_agent0.csv"
AGENT1_PER_CLASS_CSV = f"./logs/{DATE}/TEST_final_eval_perclass_agent1.csv"
B0_PER_CLASS_CSV = f"./logs/{DATE}/TEST_final_eval_perclass_B0.csv"
M1_PER_CLASS_CSV = f"./logs/{DATE}/TEST_final_eval_perclass_M1.csv"
# Directory for figures
FIG_DIR   = f"./figs/{DATE}/"
os.makedirs(FIG_DIR, exist_ok=True)
df_tr = pd.read_csv(TRAIN_CSV)
df = pd.read_csv(TEST_CSV)

# --- Helpers ---------------------------------------------------------------
EPS = 1e-12

def ece(probs: np.ndarray, correct: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(probs, bins) - 1
    e = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            acc = correct[m].mean()
            conf = probs[m].mean()
            e += (m.mean()) * abs(acc - conf)
    return float(e)


# --- Confusion-matrix helpers ----------------------------------------------
def _compute_cm(df_src: pd.DataFrame, pred_col: str, gt_col: str = "ground_truth"):
    """Return (raw_cm, row-normalized_cm, labels) as DataFrames.
    Ensures square matrix over observed class labels.
    """
    # use sorted unique labels from ground truth to define axes
    labels = np.sort(df_src[gt_col].unique())
    cm = pd.crosstab(df_src[gt_col], df_src[pred_col], rownames=["gt"], colnames=["pred"], dropna=False)
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    # row-normalize (per-class)
    denom = cm.sum(axis=1).replace(0, 1)
    cm_norm = cm.div(denom, axis=0)
    return cm, cm_norm, labels


def _plot_cm(cm_df: pd.DataFrame, title: str, outpath: str, vmax: float = 1.0):
    fig = plt.figure(figsize=(6.0, 5.5))
    ax = fig.add_subplot(111)
    sns.heatmap(cm_df, ax=ax, cmap="Blues", vmin=0.0, vmax=vmax, square=True, cbar=True)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.tick_params(which="minor", bottom=False, left=False, top=False, right=False)
    ax.tick_params(which="major", top=False, right=False)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title(title)
    # Improve tick label density (show every Nth if there are many classes)
    n = cm_df.shape[0]
    if n > 30:
        step = int(np.ceil(n / 30))
        ax.set_xticks(np.arange(0.5, n + 0.5, step))
        ax.set_yticks(np.arange(0.5, n + 0.5, step))
    fig.tight_layout()
    fig.savefig(outpath, dpi=1200, bbox_inches="tight")
    base, _ = os.path.splitext(outpath)
    fig.savefig(base + ".pdf", dpi=1200, bbox_inches="tight", format="pdf")
    plt.close(fig)

# --- 1) Training curves (accuracy only) ------------------------------------
if os.path.exists(TRAIN_CSV):


    # cumulative accuracy after round t
    df_tr["cumulative_acc_B0"] = df_tr["acc_B0"].cumsum() / (df_tr["round"] + 1)
    df_tr["cumulative_acc_M1"] = df_tr["acc_M1"].cumsum() / (df_tr["round"] + 1)

    fig1 = plt.figure(figsize=(6.0, 3.3))
    ax1 = fig1.add_subplot(111)
    ax1.plot(df_tr["round"][10:], df_tr["cumulative_acc_B0"][10:], label="Acc B0")
    ax1.plot(df_tr["round"][10:], df_tr["cumulative_acc_M1"][10:], label="Acc M1")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Training Accuracy (Cumulative)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    fig1.tight_layout()
    outpath1 = os.path.join(FIG_DIR, "train_accuracy.png")
    fig1.savefig(outpath1, dpi=1200, bbox_inches="tight")
    base1, _ = os.path.splitext(outpath1)
    fig1.savefig(base1 + ".pdf", dpi=1200, bbox_inches="tight", format="pdf")
    plt.close(fig1)
else:
    print(f"[warn] Training CSV not found: {TRAIN_CSV}")

# --- 2) Test-set summaries & plots -----------------------------------------
if not os.path.exists(TEST_CSV):
    raise FileNotFoundError(f"Test CSV not found: {TEST_CSV}")


# headline accuracies
accs = {
    "Agent0": float((df["agent0_pred"] == df["ground_truth"]).mean()),
    "Agent1": float((df["agent1_pred"] == df["ground_truth"]).mean()),
    "B0": float(df["acc_B0"].mean()),
    "M1": float(df["acc_M1"].mean()),
}
print(accs)

# McNemar M1 vs B0
m1c = df["acc_M1"] == 1
b0c = df["acc_B0"] == 1
b = int(( m1c & ~b0c ).sum())  # M1 correct, B0 wrong
c = int((~m1c &  b0c ).sum())  # M1 wrong,  B0 correct
from statsmodels.stats.contingency_tables import mcnemar
exact = True if b + c < 25 else False
print("McNemar:", mcnemar([[0, c], [b, 0]], exact=exact))

# --- 2a) Accuracy gain vs cosine-similarity (line + fill) ------------------
# Use quantile binning with duplicates dropped to avoid duplicate end-bins.
if "cos_sim_beliefs" in df:
    cos = df["cos_sim_beliefs"].astype(float)
    cats = pd.qcut(cos, q=10, duplicates='drop')  # categorical with Interval categories
    df_cos = (
        df.assign(cos_bin=cats)
          .groupby("cos_bin")[ ["acc_B0", "acc_M1"] ]
          .mean()
          .dropna()
    )
    acc_gain = (df_cos["acc_M1"] - df_cos["acc_B0"]).values
    bins = df_cos.index
    # X as bin centers for a smoother look
    x = np.array([ (iv.left + iv.right) / 2.0 for iv in bins ])

    fig2 = plt.figure(figsize=(6.0, 3.3))
    ax2 = fig2.add_subplot(111)
    ax2.plot(x, acc_gain, marker='o', linewidth=1.5)
    ax2.fill_between(x, acc_gain, 0, alpha=0.25, step=None)
    ax2.set_xlabel("Cosine similarity between agents' beliefs")
    ax2.set_ylabel("Accuracy gain (M1 - B0)")
    ax2.set_title("Fusion Gain vs. Disagreement (cosine)")
    ax2.axhline(0, color='k', linewidth=0.8, alpha=0.5)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    outpath2 = os.path.join(FIG_DIR, "cosine_gain.png")
    fig2.savefig(outpath2, dpi=1200, bbox_inches='tight')
    base2, _ = os.path.splitext(outpath2)
    fig2.savefig(base2 + ".pdf", dpi=1200, bbox_inches='tight', format="pdf")
    plt.close(fig2)
else:
    print("[warn] 'cos_sim_beliefs' not found; skipping cosine plot.")

# --- 2b) Cumulative NLL and ECE (printed; no plots by default) -------------
if {"p_b0_gt", "p_m1_gt"}.issubset(df.columns):
    df["_nll_b0"] = -np.log(df["p_b0_gt"].clip(EPS))
    df["_nll_m1"] = -np.log(df["p_m1_gt"].clip(EPS))
    df["_cum_b0"] = df["_nll_b0"].cumsum()
    df["_cum_m1"] = df["_nll_m1"].cumsum()
    print("Final cum NLL:", df[["_cum_b0", "_cum_m1"]].iloc[-1].to_dict())

if "m1_conf_top1" in df:
    e = ece(df["m1_conf_top1"].values, df["acc_M1"].astype(bool).values, n_bins=10)
    print("ECE(M1):", e)

# --- 2c) Cumulative Regret vs B0 (uses p_b0_gt, p_m1_gt) -------------------
if {"p_b0_gt", "p_m1_gt"}.issubset(df.columns):
    df["_nll_b0"] = -np.log(df["p_b0_gt"].clip(EPS))
    df["_nll_m1"] = -np.log(df["p_m1_gt"].clip(EPS))
    df["_regret_vs_b0"] = (df["_nll_m1"] - df["_nll_b0"]).cumsum()

    fig3 = plt.figure(figsize=(6.0, 3.3))
    ax3 = fig3.add_subplot(111)
    x = np.arange(len(df))
    ax3.plot(x, df["_regret_vs_b0"], label="Regret (M1 - B0)")
    ax3.axhline(0, linestyle="--", linewidth=1, color="k", alpha=0.5)
    ax3.set_xlabel("Example index (test order)")
    ax3.set_ylabel("Cumulative regret")
    ax3.set_title("Cumulative Regret vs Baseline")
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    fig3.tight_layout()
    outpath3 = os.path.join(FIG_DIR, "regret_vs_b0.png")
    fig3.savefig(outpath3, dpi=1200, bbox_inches='tight')
    base3, _ = os.path.splitext(outpath3)
    fig3.savefig(base3 + ".pdf", dpi=1200, bbox_inches='tight', format="pdf")
    plt.close(fig3)

    # Optional: regret vs oracle single agent if logged
    if {"p_a0_gt", "p_a1_gt"}.issubset(df.columns):
        df["_p_oracle"] = df[["p_a0_gt", "p_a1_gt"]].max(axis=1)
        df["_nll_oracle"] = -np.log(df["_p_oracle"].clip(EPS))
        df["_regret_vs_oracle"] = (df["_nll_m1"] - df["_nll_oracle"]).cumsum()

        fig3b = plt.figure(figsize=(6.0, 3.3))
        ax3b = fig3b.add_subplot(111)
        ax3b.plot(x, df["_regret_vs_oracle"], label="Regret (M1 - Oracle agent)")
        ax3b.axhline(0, linestyle="--", linewidth=1, color="k", alpha=0.5)
        ax3b.set_xlabel("Example index (test order)")
        ax3b.set_ylabel("Cumulative regret")
        ax3b.set_title("Cumulative Regret vs Oracle Single Agent")
        ax3b.legend()
        ax3b.grid(True, alpha=0.3)
        fig3b.tight_layout()
        outpath3b = os.path.join(FIG_DIR, "regret_vs_oracle.png")
        fig3b.savefig(outpath3b, dpi=1200, bbox_inches='tight')
        base3b, _ = os.path.splitext(outpath3b)
        fig3b.savefig(base3b + ".pdf", dpi=1200, bbox_inches='tight', format="pdf")
        plt.close(fig3b)

# --- 2d) Correlation: Hedge vs Attention weights ---------------------------
if {"hedge0", "hedge1", "pi0", "pi1"}.issubset(df_tr.columns):
    df_tr["_d_hedge"] = df_tr["hedge1"].astype(float) - df_tr["hedge0"].astype(float)
    df_tr["_d_pi"]    = df_tr["pi1"].astype(float)    - df_tr["pi0"].astype(float)

    # Pearson & Spearman correlations (printed)
    pearson_r, pearson_p = stats.pearsonr(df_tr["_d_hedge"], df_tr["_d_pi"]) if len(df_tr) > 1 else (np.nan, np.nan)
    spearman_rho, spearman_p = stats.spearmanr(df_tr["_d_hedge"], df_tr["_d_pi"], nan_policy="omit") if len(df_tr) > 1 else (np.nan, np.nan)
    print(f"Hedge↔Attention Δ correlation: Pearson r={pearson_r:.3f} (p={pearson_p:.2e}), Spearman ρ={spearman_rho:.3f} (p={spearman_p:.2e})")

    # Scatter with trend line
    fig4 = plt.figure(figsize=(6.0, 3.3))
    ax4 = fig4.add_subplot(111)
    ax4.scatter(df_tr["_d_hedge"], df_tr["_d_pi"], s=8, alpha=0.4)
    if df_tr["_d_hedge"].notna().sum() >= 2:
        m, b = np.polyfit(df_tr["_d_hedge"], df_tr["_d_pi"], 1)
        xs = np.linspace(df_tr["_d_hedge"].min(), df_tr["_d_hedge"].max(), 100)
        ax4.plot(xs, m*xs + b, linewidth=1)
    ax4.axhline(0, linestyle="--", linewidth=1, color="k", alpha=0.5)
    ax4.axvline(0, linestyle="--", linewidth=1, color="k", alpha=0.5)
    ax4.set_xlabel(r"$\Delta$ hedge (hedge1 - hedge0)")
    ax4.set_ylabel(r"$\Delta$ attention (pi1 - pi0)")
    ax4.set_title("Alignment of Hedge and Attention")
    ax4.grid(True, alpha=0.3)
    fig4.tight_layout()
    outpath4 = os.path.join(FIG_DIR, "hedge_attention_alignment.png")
    fig4.savefig(outpath4, dpi=1200, bbox_inches='tight')
    base4, _ = os.path.splitext(outpath4)
    fig4.savefig(base4 + ".pdf", dpi=1200, bbox_inches='tight', format="pdf")
    plt.close(fig4)

    # Strength (magnitudes)
    fig5 = plt.figure(figsize=(6.0, 3.3))
    ax5 = fig5.add_subplot(111)
    abs_h = np.abs(df_tr["_d_hedge"]).values
    abs_p = np.abs(df_tr["_d_pi"]).values
    ax5.scatter(abs_h, abs_p, s=8, alpha=0.4)
    if np.isfinite(abs_h).sum() >= 2:
        m2, b2 = np.polyfit(abs_h, abs_p, 1)
        xs2 = np.linspace(np.nanmin(abs_h), np.nanmax(abs_h), 100)
        ax5.plot(xs2, m2*xs2 + b2, linewidth=1)
    ax5.set_xlabel(r"|$\Delta$ hedge|")
    ax5.set_ylabel(r"|$\Delta$ attention|")
    ax5.set_title("Strength Alignment (Magnitudes)")
    ax5.grid(True, alpha=0.3)
    fig5.tight_layout()
    outpath5 = os.path.join(FIG_DIR, "hedge_attention_strength.png")
    fig5.savefig(outpath5, dpi=1200, bbox_inches='tight')
    base5, _ = os.path.splitext(outpath5)
    fig5.savefig(base5 + ".pdf", dpi=1200, bbox_inches='tight', format="pdf")
    plt.close(fig5)
else:
    print("[warn] Missing columns for hedge/attention correlation; skipping.")


# --- 2e) Confusion matrices (row-normalized + raw counts) -------------------
if {"ground_truth", "agent0_pred", "agent1_pred", "fused_B0_pred", "fused_M1_pred"}.issubset(df.columns):
    cm_specs = [
        ("agent0_pred", "Agent0 (Vision)"),
        ("agent1_pred", "Agent1 (Audio)"),
        ("fused_B0_pred", "B0 (Baseline Fusion)"),
        ("fused_M1_pred", "M1 (Attention + Hedge)")
    ]
    for col, name in cm_specs:
        cm_raw, cm_norm, _ = _compute_cm(df, pred_col=col, gt_col="ground_truth")
        outpath_norm = os.path.join(FIG_DIR, f"confmat_{col}_norm.png")
        _plot_cm(cm_norm, f"Confusion Matrix (normalized) — {name}", outpath_norm, vmax=1.0)
        outpath_raw = os.path.join(FIG_DIR, f"confmat_{col}_raw.png")
        vmax_val = cm_raw.values.max() if cm_raw.values.max() > 0 else 1.0
        _plot_cm(cm_raw,  f"Confusion Matrix (counts) — {name}", outpath_raw,  vmax=vmax_val)
else:
    print("[warn] Missing columns for confusion matrices; skipping.")

# --- 2f) Per-class F1 comparison (three-way split by unimodal weakness) ---
try:
    have_all = all(os.path.exists(p) for p in [
        AGENT0_PER_CLASS_CSV, AGENT1_PER_CLASS_CSV, B0_PER_CLASS_CSV, M1_PER_CLASS_CSV
    ])
    if not have_all:
        missing = [p for p in [AGENT0_PER_CLASS_CSV, AGENT1_PER_CLASS_CSV, B0_PER_CLASS_CSV, M1_PER_CLASS_CSV] if not os.path.exists(p)]
        print(f"[warn] Missing per-class CSV(s): {missing} — skipping per-class F1 plots.")
    else:
        # Load per-class metrics
        a0 = pd.read_csv(AGENT0_PER_CLASS_CSV)
        a1 = pd.read_csv(AGENT1_PER_CLASS_CSV)
        b0 = pd.read_csv(B0_PER_CLASS_CSV)
        m1 = pd.read_csv(M1_PER_CLASS_CSV)

        # Standardize columns and modality labels
        def _prep(df_in, modality_name):
            df_out = df_in.rename(columns={"class": "class_id"})
            keep = ["class_id", "f1"]
            df_out = df_out[keep].copy()
            df_out["Modality"] = modality_name
            return df_out

        df_list = [
            _prep(a0, "Agent0"),
            _prep(a1, "Agent1"),
            _prep(b0, "B0"),
            _prep(m1, "M1"),
        ]
        df_all = pd.concat(df_list, ignore_index=True)

        # Wide pivot: rows=class_id, cols=Modality, values=F1
        f1_wide = df_all.pivot_table(index="class_id", columns="Modality", values="f1", aggfunc="mean")
        f1_wide_filled = f1_wide.fillna(0.0)

        # Threshold for poor performance
        THR = 0.5
        poor_a0 = f1_wide_filled.get("Agent0", pd.Series(0, index=f1_wide_filled.index)) < THR
        poor_a1 = f1_wide_filled.get("Agent1", pd.Series(0, index=f1_wide_filled.index)) < THR

        both_poor_idx   = f1_wide_filled.index[ poor_a0 &  poor_a1 ]
        vision_poor_idx = f1_wide_filled.index[ poor_a0 & ~poor_a1 ]
        audio_poor_idx  = f1_wide_filled.index[~poor_a0 &  poor_a1 ]

        def _plot_bucket(indices, title, fname):
            if len(indices) == 0:
                print(f"[info] No classes for bucket: {title}; skipping plot.")
                return
            cols_present = [c for c in ["Agent0","Agent1","B0","M1"] if c in f1_wide.columns]
            f1_long = (
                f1_wide.loc[indices, cols_present]
                .reset_index()
                .melt(id_vars="class_id", var_name="Modality", value_name="F1")
            )
            # Order by min unimodal F1 to surface the hardest classes first
            if all(c in f1_wide_filled.columns for c in ["Agent0","Agent1"]):
                order_score = np.minimum(
                    f1_wide_filled.loc[indices, "Agent0"].values,
                    f1_wide_filled.loc[indices, "Agent1"].values
                )
                order = list(np.array(indices)[np.argsort(order_score)])
                f1_long["class_id"] = pd.Categorical(f1_long["class_id"], categories=order, ordered=True)
                f1_long = f1_long.sort_values(["class_id","Modality"]).reset_index(drop=True)

            fig = plt.figure(figsize=(max(6.5, 0.45*len(indices) + 2.5), 3.6))
            ax = fig.add_subplot(111)
            sns.barplot(data=f1_long, x="class_id", y="F1", hue="Modality", ax=ax)
            ax.set_xlabel("Class ID")
            ax.set_ylabel("F1 score")
            ax.set_title(title)
            ax.set_ylim(0, 1.0)
            ax.legend(frameon=False, ncol=4, loc="upper right")
            ax.tick_params(axis="x", rotation=90)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            outpng = os.path.join(FIG_DIR, f"{fname}.png")
            fig.savefig(outpng, dpi=1200, bbox_inches="tight")
            base, _ = os.path.splitext(outpng)
            fig.savefig(base + ".pdf", dpi=1200, bbox_inches="tight", format="pdf")
            plt.close(fig)

        _plot_bucket(both_poor_idx,   "Per-class F1 (both unimodal poor)",   "perclass_f1_both_poor")
        _plot_bucket(vision_poor_idx, "Per-class F1 (vision poor, audio ok)", "perclass_f1_vision_poor")
        _plot_bucket(audio_poor_idx,  "Per-class F1 (audio poor, vision ok)",  "perclass_f1_audio_poor")
except Exception as e:
    print(f"[error] per-class F1 three-way plotting failed: {e}")