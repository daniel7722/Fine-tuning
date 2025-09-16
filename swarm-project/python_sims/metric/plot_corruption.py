import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns


# --- Global style -----------------------------------------------------------
plt.style.use("science")
plt.rcParams.update({
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "legend.fontsize": 12,
})
sns.set_theme(context="paper", style=None, rc=plt.rcParams)
# ---------- Helpers ----------

def read_main_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Try to normalize column names if needed
    cols = {c.lower(): c for c in df.columns}
    # Minimal required fields
    # idx might be called 'round' or similar; fall back to range if missing
    if 'idx' not in cols:
        df['idx'] = np.arange(len(df))
    if 'acc_b0' not in cols and 'acc_b0' not in df.columns:
        # Try to reconstruct if not present (rare)
        if 'fused_b0_pred' in df.columns and 'gt' in df.columns:
            df['acc_b0'] = (df['fused_b0_pred'] == df['gt']).astype(int)
        else:
            df['acc_b0'] = np.nan
    if 'acc_m1' not in df.columns and 'fused_m1_pred' in df.columns and 'gt' in df.columns:
        df['acc_m1'] = (df['fused_m1_pred'] == df['gt']).astype(int)
    # Loss columns may be strings in some logs; coerce to numeric
    for c in ['loss_b0', 'loss_m1', 'p_b0_gt', 'p_m1_gt', 'm1_conf_top1', 'm1_margin']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    # Hedge weights packed? If present, they might be arrays; we prefer extras csv for these
    return df

def read_extras_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure numeric types
    for c in ['is_corrupt','b0_conf','ent_agent0','ent_agent1',
              'log_odds_audio_vs_vision','loss_agent0','loss_agent1']:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df

def rolling_mean(x, w=25):
    return pd.Series(x).rolling(w, min_periods=max(1, w//5)).mean().to_numpy()

def shade_corruption(ax, idx, is_corrupt, color=None, alpha=0.15):
    # Shade contiguous corrupt segments
    in_seg = False
    start = None
    for i, flag in zip(idx, is_corrupt):
        if flag and not in_seg:
            in_seg = True
            start = i
        if in_seg and not flag:
            ax.axvspan(start, i, alpha=alpha, color=color)
            in_seg = False
    if in_seg:
        ax.axvspan(start, idx[-1], alpha=alpha, color=color)

# ---------- Plotting for one run ----------

def plot_run(run_name: str, main_csv: Path, extras_csv: Path, outdir: Path, w=25, corrupted_mod: str | None = None):
    df = read_main_csv(main_csv)
    ex = read_extras_csv(extras_csv)

    # Align on idx (outer join to be safe)
    key = 'idx'
    if key not in df.columns:
        df[key] = np.arange(len(df))
    if key not in ex.columns:
        ex[key] = np.arange(len(ex))
    m = pd.merge(df, ex, on=key, how='inner', suffixes=('', '_ex'))

    # Basic series
    idx = m[key].to_numpy()
    is_corrupt = m['is_corrupt'].fillna(0).to_numpy().astype(int)

    # 1) Hedge weights proxy: using log-odds -> convert to weight if you want
    #   w_audio = sigmoid(log_odds), w_vision = 1 - w_audio
    if 'log_odds_audio_vs_vision' in m.columns:
        log_odds = m['log_odds_audio_vs_vision'].to_numpy()
        w_audio = 1.0 / (1.0 + np.exp(-log_odds))
        w_vision = 1.0 - w_audio
    else:
        w_audio = w_vision = None

    # 2) Rolling accuracies
    acc_b0 = m.get('acc_B0', pd.Series(np.nan, index=m.index)).to_numpy()
    # print(acc_b0)
    acc_m1 = m.get('acc_M1', pd.Series(np.nan, index=m.index)).to_numpy()
    acc_b0_roll = rolling_mean(acc_b0, w=w)
    # print(acc_b0_roll)
    acc_m1_roll = rolling_mean(acc_m1, w=w)

    # 3) Rolling NLL
    loss_b0 = m.get('loss_b0', pd.Series(np.nan, index=m.index)).to_numpy()
    loss_m1 = m.get('loss_m1', pd.Series(np.nan, index=m.index)).to_numpy()
    loss_b0_roll = rolling_mean(loss_b0, w=w)
    loss_m1_roll = rolling_mean(loss_m1, w=w)

    # 4) Entropies
    ent0 = m.get('ent_agent0', pd.Series(np.nan, index=m.index)).to_numpy()
    ent1 = m.get('ent_agent1', pd.Series(np.nan, index=m.index)).to_numpy()
    ent0_roll = rolling_mean(ent0, w=w)
    ent1_roll = rolling_mean(ent1, w=w)


    # --- Figure 1: Hedge weights over time ---
    fig, ax = plt.subplots(figsize=(10, 4))
    if w_audio is not None:
        ax.plot(idx, w_audio, label='w_audio (from log-odds)')
        ax.plot(idx, w_vision, label='w_vision')
    else:
        ax.text(0.5, 0.5, 'No log-odds in extras CSV', transform=ax.transAxes, ha='center')
    shade_corruption(ax, idx, is_corrupt)
    ax.set_title('Hedge weights over time', fontsize=14)
    ax.set_xlabel('Index', fontsize=14)
    ax.set_ylabel('Weight', fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f'{run_name}_hedge_weights.pdf', dpi=1200)
    plt.close(fig)

    # --- Figure 2: Rolling accuracy ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, acc_b0_roll, label='B0 rolling acc')
    ax.plot(idx, acc_m1_roll, label='M1 rolling acc')
    shade_corruption(ax, idx, is_corrupt)
    ax.set_title(f'Rolling accuracy (window={w})', fontsize=14)
    ax.set_xlabel('Index', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f'{run_name}_rolling_accuracy.pdf', dpi=1200)
    plt.close(fig)

    # --- Figure 3: Rolling NLL ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, loss_b0_roll, label='B0 rolling NLL')
    ax.plot(idx, loss_m1_roll, label='M1 rolling NLL')
    shade_corruption(ax, idx, is_corrupt)
    ax.set_title(f'Rolling NLL (window={w})', fontsize=14)
    ax.set_xlabel('Index', fontsize=14)
    ax.set_ylabel('NLL', fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f'{run_name}_rolling_nll.pdf', dpi=1200)
    plt.close(fig)

    # --- Figure 4: Agent entropies ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, ent0_roll, label='Entropy agent0')
    ax.plot(idx, ent1_roll, label='Entropy agent1')
    shade_corruption(ax, idx, is_corrupt)
    ax.set_title(f'Agent belief entropy (window={w})', fontsize=14)
    ax.set_xlabel('Index', fontsize=14)
    ax.set_ylabel('Entropy (nats)', fontsize=14)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f'{run_name}_entropies.pdf', dpi=1200)
    plt.close(fig)

    # --- Figure 4b: Attention weights over time (if logged) ---
    # Accept columns named either 'pi_audio'/'pi_vision' or 'pi1'/'pi0'
    pi_audio_series = None
    pi_vision_series = None
    if 'pi_audio' in m.columns and 'pi_vision' in m.columns:
        pi_audio_series = pd.to_numeric(m['pi_audio'], errors='coerce')
        pi_vision_series = pd.to_numeric(m['pi_vision'], errors='coerce')
    elif 'pi1' in m.columns and 'pi0' in m.columns:
        pi_audio_series = pd.to_numeric(m['pi1'], errors='coerce')
        pi_vision_series = pd.to_numeric(m['pi0'], errors='coerce')

    if pi_audio_series is not None and pi_vision_series is not None:
        pi_audio = pi_audio_series.to_numpy()
        pi_vision = pi_vision_series.to_numpy()
        pi_audio_roll = rolling_mean(pi_audio, w=w)
        pi_vision_roll = rolling_mean(pi_vision, w=w)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(idx, pi_audio_roll, label='$\pi$_audio (roll)')
        ax.plot(idx, pi_vision_roll, label='$\pi$_vision (roll)')
        shade_corruption(ax, idx, is_corrupt)
        ax.set_title(f'Attention weights (window={w})', fontsize=14)
        ax.set_xlabel('Index', fontsize=14)
        ax.set_ylabel('Attention weight')
        ax.set_ylim(-0.05, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(outdir / f'{run_name}_attn_weights.pdf', dpi=1200)
        plt.close(fig)

        # If corrupted modality specified, run gate analyses
        if corrupted_mod in ('audio','vision'):
            pi_corr = pi_audio if corrupted_mod=='audio' else pi_vision
            inside = is_corrupt == 1
            outside = is_corrupt == 0
            mean_in = float(np.nanmean(pi_corr[inside])) if np.any(inside) else np.nan
            mean_out = float(np.nanmean(pi_corr[outside])) if np.any(outside) else np.nan
            delta = mean_in - mean_out
            # t-test if scipy available
            try:
                import scipy.stats as st
                tstat, pval = st.ttest_ind(pi_corr[inside], pi_corr[outside], equal_var=False, nan_policy='omit')
            except Exception:
                tstat, pval = np.nan, np.nan
            print(f"[GATE] {run_name}: $\pi$_{corrupted_mod} IN={mean_in:.3f} OUT={mean_out:.3f} $\delta$={delta:.3f} p={pval:.2e}")

            # Boxplot clean vs corrupt for π_corrupt
            try:
                fig, ax = plt.subplots(figsize=(5, 3.5))
                data = [pi_corr[outside], pi_corr[inside]]
                ax.boxplot(data, labels=['clean','corrupt'], showmeans=True)
                ax.set_ylabel(f"π_{corrupted_mod}")
                ax.set_title('Gate response to corruption')
                fig.tight_layout()
                fig.savefig(outdir / f'{run_name}_gate_box.pdf', dpi=1200)
                plt.close(fig)
            except Exception:
                pass

            # AUROC using 1 - π_corrupt as score (higher means more likely corrupt)
            try:
                from sklearn.metrics import roc_auc_score
                y = is_corrupt
                score = 1.0 - pi_corr
                # Need finite values only
                mask = np.isfinite(score) & np.isfinite(y)
                auc = roc_auc_score(y[mask], score[mask])
                print(f"[GATE] {run_name}: AUROC(1-$\pi$_{corrupted_mod} to is_corrupt) = {auc:.3f}")
            except Exception as e:
                print(f"[GATE] {run_name}: AUROC unavailable ({e})")

    # 5) Top-1 confidence (both B0 and M1) with same rolling window
    b0_conf = m.get('b0_conf', pd.Series(np.nan, index=m.index)).to_numpy()
    b0_conf_roll = rolling_mean(b0_conf, w=w)

    m1_conf = m.get('m1_conf_top1', pd.Series(np.nan, index=m.index)).to_numpy()
    m1_conf_roll = rolling_mean(m1_conf, w=w)

    # --- Figure 5: Confidence ---
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(idx, b0_conf_roll, label='B0 top-1 conf')
    ax.plot(idx, m1_conf_roll, label='M1 top-1 conf')
    shade_corruption(ax, idx, is_corrupt)
    ax.set_title(f'Confidence (window={w})', fontsize=14)
    ax.set_xlabel('Index', fontsize=14)
    ax.set_ylabel('Top-1 confidence', fontsize=14)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / f'{run_name}_confidence.pdf', dpi=1200)
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--date', required=True, help='DATE folder under logs/')
    ap.add_argument('--runs', nargs='+', required=True,
                    help='Base run names (log_filename). For each X, expects X.csv and X_corruption_signals.csv')
    ap.add_argument('--window', type=int, default=25, help='Rolling window size')
    ap.add_argument('--corrupted_mod', choices=['audio','vision'], default=None, help='If set, indicates which modality was corrupted for this run; enables gate analyses.')
    args = ap.parse_args()

    outdir = Path('figs') / args.date
    outdir.mkdir(parents=True, exist_ok=True)
    base = Path('logs') / args.date

    for run_name in args.runs:
        main_csv = base / f'{run_name}.csv'
        extras_csv = base / f'{run_name}_corruption_signals.csv'
        if not main_csv.exists():
            print(f'[WARN] Missing main CSV: {main_csv}')
            continue
        if not extras_csv.exists():
            print(f'[WARN] Missing extras CSV: {extras_csv}')
            continue
        print(f'[PLOT] {run_name}')
        plot_run(run_name, main_csv, extras_csv, outdir, w=args.window, corrupted_mod=args.corrupted_mod)

if __name__ == '__main__':
    main()