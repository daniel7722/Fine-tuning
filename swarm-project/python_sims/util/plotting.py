

"""
Plotting utilities for AVE simulation logs.

Generates:
- Per-class accuracy heatmaps (per round or for the final round)
- Round-by-round curves for overall accuracy, loss, and hedge weights

Usage examples:

  # Heatmap for the final round (default)
  python -m python_sims.util.plotting \
      --log_dir logs/2025-08-12 \
      --perclass_csv AVE_perclass.csv \
      --heatmap out/heatmap_last.png

  # Heatmap for a specific round
  python -m python_sims.util.plotting \
      --log_dir logs/2025-08-12 \
      --perclass_csv AVE_perclass.csv \
      --round 1200 \
      --heatmap out/heatmap_r1200.png

  # Time-series plots from the main log
  python -m python_sims.util.plotting \
      --log_dir logs/2025-08-12 \
      --main_csv AVE_run.csv \
      --curves out/curves.png \
      --hedge out/hedge.png

  # Per-class accuracy over time for one class (id=7)
  python -m python_sims.util.plotting \
      --log_dir logs/2025-08-12 \
      --perclass_csv AVE_perclass.csv \
      --class_curve 7 out/class7_curve.png

"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Helpers
# -----------------------------

def _resolve(path_or_name: str, base: Path) -> Path:
    p = Path(path_or_name)
    return p if p.is_absolute() else (base / p)


def _ensure_out(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Per-class heatmap(s)
# -----------------------------

def plot_heatmap(perclass_csv: Path, out_png: Path, round_idx: Optional[int] = None) -> None:
    """Plot a per-class accuracy heatmap for a given round, or the final round if None."""
    df = pd.read_csv(perclass_csv)
    if round_idx is None:
        # Pick the max round available per modality/class
        round_idx = int(df["round"].max())

    dfr = df[df["round"] == round_idx].copy()
    if dfr.empty:
        raise ValueError(f"No rows found for round={round_idx} in {perclass_csv}")

    # pivot: class x modality -> acc
    pivot = dfr.pivot_table(index="class_id", columns="modality", values="acc", aggfunc="first")
    # Sort classes by fusion (if present) otherwise by mean acc
    if "fusion" in pivot.columns:
        pivot = pivot.sort_values(by="fusion", ascending=False)
    else:
        pivot["_mean"] = pivot.mean(axis=1)
        pivot = pivot.sort_values(by="_mean", ascending=False).drop(columns=["_mean"])  # type: ignore

    # plot
    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.25)))
    im = ax.imshow(pivot.values, aspect="auto", interpolation="nearest")
    ax.set_title(f"Per-class accuracy @ round {round_idx}")
    ax.set_xlabel("Modality")
    ax.set_ylabel("Class ID (sorted)")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    # colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Accuracy")

    _ensure_out(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Main log curves (accuracy/loss & hedge weights)
# -----------------------------

def _parse_correct_pct(series: pd.Series) -> pd.Series:
    # series like '63.04%' -> float 63.04
    return series.astype(str).str.rstrip('%').astype(float)


def plot_curves(main_csv: Path, out_png: Path) -> None:
    """Plot overall accuracy (%) and loss over rounds."""
    df = pd.read_csv(main_csv)
    if df.empty:
        raise ValueError(f"No data in {main_csv}")

    rounds = df["round"].to_numpy()
    acc = _parse_correct_pct(df["correct_percentage"]).to_numpy()
    loss = df["loss"].astype(float).to_numpy()

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(rounds, acc, label="Accuracy (%)")
    ax1.set_xlabel("Round")
    ax1.set_ylabel("Accuracy (%)")

    ax2 = ax1.twinx()
    ax2.plot(rounds, loss, label="Loss", linestyle="--")
    ax2.set_ylabel("Loss")

    lines, labels = [], []
    for ax in (ax1, ax2):
        L, lab = ax.get_legend_handles_labels()
        lines += L; labels += lab
    ax1.legend(lines, labels, loc="best")

    _ensure_out(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def plot_hedge(main_csv: Path, out_png: Path) -> None:
    """Plot hedge weights over rounds for agent0 / agent1."""
    df = pd.read_csv(main_csv)
    if df.empty:
        raise ValueError(f"No data in {main_csv}")

    rounds = df["round"].to_numpy()
    a0 = df["agent0_hw"].astype(float).to_numpy()
    a1 = df["agent1_hw"].astype(float).to_numpy()

    fig, ax = plt.subplots(figsize=(9, 3.5))
    ax.plot(rounds, a0, label="agent0 (vision?)")
    ax.plot(rounds, a1, label="agent1 (audio?)")
    ax.set_xlabel("Round")
    ax.set_ylabel("Hedge weight")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")

    _ensure_out(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# Per-class accuracy curve for one class
# -----------------------------

def plot_class_curve(perclass_csv: Path, class_id: int, out_png: Path) -> None:
    df = pd.read_csv(perclass_csv)
    dfc = df[df["class_id"] == class_id].copy()
    if dfc.empty:
        raise ValueError(f"No rows for class_id={class_id} in {perclass_csv}")

    # Modality lines over rounds
    fig, ax = plt.subplots(figsize=(9, 4))
    for mod, g in dfc.groupby("modality"):
        ax.plot(g["round"], g["acc"], label=str(mod))
    ax.set_title(f"Per-class accuracy over time (class {class_id})")
    ax.set_xlabel("Round")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1)
    ax.legend(loc="best")

    _ensure_out(out_png)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description="Plot AVE simulation logs")
    p.add_argument("--log_dir", type=str, required=True, help="Directory containing CSV logs")
    p.add_argument("--perclass_csv", type=str, default="AVE_perclass.csv")
    p.add_argument("--main_csv", type=str, default="AVE_run.csv")

    # Outputs (optional)
    p.add_argument("--heatmap", type=str, help="Output PNG for per-class heatmap")
    p.add_argument("--round", type=int, default=None, help="Round to plot for heatmap (default: last)")
    p.add_argument("--curves", type=str, help="Output PNG for accuracy/loss curves")
    p.add_argument("--hedge", type=str, help="Output PNG for hedge weight curves")

    # Class-specific curve
    p.add_argument("--class_curve", nargs=2, metavar=("CLASS_ID", "OUT_PNG"), help="Plot per-class accuracy over rounds")

    args = p.parse_args()
    base = Path(args.log_dir)

    # Heatmap
    if args.heatmap:
        plot_heatmap(
            perclass_csv=_resolve(args.perclass_csv, base),
            out_png=_resolve(args.heatmap, base),
            round_idx=args.round,
        )

    # Curves
    if args.curves:
        plot_curves(
            main_csv=_resolve(args.main_csv, base),
            out_png=_resolve(args.curves, base),
        )

    # Hedge
    if args.hedge:
        plot_hedge(
            main_csv=_resolve(args.main_csv, base),
            out_png=_resolve(args.hedge, base),
        )

    # Class curve
    if args.class_curve:
        class_id = int(args.class_curve[0])
        out_path = _resolve(args.class_curve[1], base)
        plot_class_curve(_resolve(args.perclass_csv, base), class_id, out_path)


if __name__ == "__main__":
    main()