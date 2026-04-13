"""Plot training curves from LLM fine-tuning metrics CSVs.

Reads ``metrics.csv`` from :func:`finetune_llm_sft`, :func:`finetune_llm_preference`,
or :func:`finetune_llm_reasoning` (population-metric columns from ``train_llm``).

Styling uses ``demos/plot_style.py`` (AgileRL palette + Inter).

Usage::

    python demos/plot_llm_metrics.py outputs/run/metrics.csv -o figures/
    python demos/plot_llm_metrics.py outputs/run/metrics.csv --algo SFT --smoothing 0.85
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from agilerl.demos.plot_style import (
    ACCENT_ORANGE,
    LIGHT_GREEN,
    PALETTE,
    PRIMARY,
    legend_kw,
    register_inter_font,
    style_axes,
)

LOSS_RAW = LIGHT_GREEN
LOSS_SMOOTH = PRIMARY
REWARD_CHOSEN = PRIMARY
REWARD_REJECTED = LIGHT_GREEN
REWARD_MARGIN = ACCENT_ORANGE

LOSS_COL = "Train/Mean population loss"
CHOSEN_COL = "Train/Mean population chosen reward"
REJECTED_COL = "Train/Mean population rejected reward"
MARGIN_COL = "Train/Mean population reward margin"


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
    return smoothed


def _build_title(algo: str, plot_type: str, params: str | None = None) -> str:
    title = f"{algo} {plot_type}"
    if params:
        title += f"    ({params})"
    return title


def plot_training_loss(
    df: pd.DataFrame,
    *,
    algo: str = "DPO",
    params: str | None = None,
    smoothing: float = 0.8,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot training loss vs step index (one CSV row per logged batch)."""
    register_inter_font()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    style_axes(ax)

    steps = np.arange(len(df))
    loss = df[LOSS_COL].values

    ax.plot(steps, loss, color=LOSS_RAW, alpha=0.55, linewidth=0.8)

    if smoothing > 0:
        log_loss = np.log(np.clip(loss, 1e-12, None))
        smoothed = np.exp(_ema(log_loss, smoothing))
        ax.plot(
            steps,
            smoothed,
            color=LOSS_SMOOTH,
            linewidth=2,
            label="Training loss (smoothed)",
        )
    else:
        ax.plot(steps, loss, color=LOSS_SMOOTH, linewidth=1.5, label="Training loss")

    ax.set_yscale("log")
    ax.set_xlabel(
        "Training step", fontsize=11, color=PALETTE["dark_grey"], fontfamily="Inter"
    )
    ax.set_ylabel("Loss", fontsize=11, color=PALETTE["dark_grey"], fontfamily="Inter")
    ax.set_title(
        _build_title(algo, "Training Loss", params),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["off_black"],
        fontfamily="Inter",
    )
    ax.legend(**legend_kw())

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["white"])
        print(f"Saved: {save_path}")
    return fig


def plot_reward_margins(
    df: pd.DataFrame,
    *,
    algo: str = "DPO",
    params: str | None = None,
    smoothing: float = 0.8,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot chosen / rejected rewards and margin (DPO CSVs)."""
    register_inter_font()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    style_axes(ax)

    steps = np.arange(len(df))
    for col, label, color in [
        (CHOSEN_COL, "Chosen reward", REWARD_CHOSEN),
        (REJECTED_COL, "Rejected reward", REWARD_REJECTED),
        (MARGIN_COL, "Reward margin", REWARD_MARGIN),
    ]:
        raw = df[col].values
        ax.plot(steps, raw, color=color, alpha=0.22, linewidth=0.7)
        if smoothing > 0:
            ax.plot(steps, _ema(raw, smoothing), color=color, linewidth=2, label=label)
        else:
            ax.plot(steps, raw, color=color, linewidth=1.5, label=label)

    ax.axhline(0, color=PALETTE["mid_grey"], linewidth=0.6, linestyle="--")
    ax.set_xlabel(
        "Training step", fontsize=11, color=PALETTE["dark_grey"], fontfamily="Inter"
    )
    ax.set_ylabel("Reward", fontsize=11, color=PALETTE["dark_grey"], fontfamily="Inter")
    ax.set_title(
        _build_title(algo, "Reward Margins", params),
        fontsize=13,
        fontweight="bold",
        color=PALETTE["off_black"],
        fontfamily="Inter",
    )
    ax.legend(**legend_kw())

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor=PALETTE["white"])
        print(f"Saved: {save_path}")
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training curves from LLM fine-tuning metrics CSVs.",
    )
    parser.add_argument("csv_path", type=str, help="Path to metrics.csv.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory for PNGs (default: same folder as the CSV).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="DPO",
        help="Title / filename label (e.g. SFT, DPO, GRPO). Default: DPO",
    )
    parser.add_argument(
        "--nll",
        action="store_true",
        help="Append '+ NLL' to the algorithm label.",
    )
    parser.add_argument("--alpha", type=float, default=None, help="Show α=… in title.")
    parser.add_argument("--beta", type=float, default=None, help="Show β=… in title.")
    parser.add_argument("--smoothing", type=float, default=0.8, help="EMA smoothing.")
    parser.add_argument(
        "--show", action="store_true", help="Show figures interactively."
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help="PNG filename prefix (default: derived from --algo).",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    algo = args.algo + (" + NLL" if args.nll else "")
    file_prefix = args.basename or algo.lower().replace(" + ", "_plus_").replace(
        " ", "_"
    )

    param_parts = []
    if args.beta is not None:
        param_parts.append(f"β={args.beta:g}")
    if args.alpha is not None:
        param_parts.append(f"α={args.alpha:g}")
    params = ", ".join(param_parts) or None

    is_dpo_reward = CHOSEN_COL in df.columns

    plot_training_loss(
        df,
        algo=algo,
        params=params,
        smoothing=args.smoothing,
        save_path=str(output_dir / f"{file_prefix}_training_loss.png"),
    )

    if is_dpo_reward:
        plot_reward_margins(
            df,
            algo=algo,
            params=params,
            smoothing=args.smoothing,
            save_path=str(output_dir / f"{file_prefix}_reward_margins.png"),
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
