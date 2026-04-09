"""Plot training curves from LLM fine-tuning metrics CSVs.

Reads the ``metrics.csv`` produced by :func:`finetune_llm_sft` or
:func:`finetune_llm_preference` and produces publication-quality loss and
reward-margin plots.

Styling uses :mod:`agilerl.utils.plot_style` (AgileRL palette + Inter from the
user cache, e.g. ``~/.cache/agilerl/fonts/inter``).

Usage::

    python demos/plot_llm_metrics.py saved_lora/20260402_093644/metrics.csv
    python demos/plot_llm_metrics.py outputs/SFT/metrics.csv -o docs/tutorials/llm_finetuning/images
    python demos/plot_llm_metrics.py saved_lora/run1/metrics.csv --smoothing 0.85
    python demos/plot_llm_metrics.py saved_lora/run1/metrics.csv --algo DPO
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

# Loss: light series + brand teal (primary) smoothed; rewards: primary / light / orange
LOSS_RAW = LIGHT_GREEN
LOSS_SMOOTH = PRIMARY
REWARD_CHOSEN = PRIMARY
REWARD_REJECTED = LIGHT_GREEN
REWARD_MARGIN = ACCENT_ORANGE


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average for smoothing noisy training curves."""
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
    return smoothed


def _build_title(algo: str, plot_type: str, params: str | None = None) -> str:
    """Compose a plot title from algorithm name, plot type, and optional params."""
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
    """Plot the training loss curve.

    :param df: DataFrame with at least ``step`` and ``train_loss`` columns.
    :param algo: Algorithm name for the title.
    :param params: Optional parameter string shown after the title (e.g. "α=1.0, β=0.1").
    :param smoothing: EMA smoothing factor (0 = none, close to 1 = heavy).
    :param save_path: If provided, save the figure to this path.
    :return: The matplotlib Figure.
    """
    register_inter_font()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    style_axes(ax)

    steps = df["step"].values
    loss = df["train_loss"].values

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
        ax.plot(
            steps,
            loss,
            color=LOSS_SMOOTH,
            linewidth=1.5,
            label="Training loss",
        )

    ax.set_yscale("log")
    ax.set_xlabel(
        "Training samples", fontsize=11, color=PALETTE["dark_grey"], fontfamily="Inter"
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
    """Plot chosen / rejected rewards and their margin over training.

    :param df: DataFrame with ``step``, ``train_chosen_reward``,
        ``train_rejected_reward``, and ``train_reward_margin`` columns.
    :param algo: Algorithm name for the title.
    :param params: Optional parameter string shown after the title.
    :param smoothing: EMA smoothing factor.
    :param save_path: If provided, save the figure to this path.
    :return: The matplotlib Figure.
    """
    register_inter_font()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    style_axes(ax)

    steps = df["step"].values

    for col, label, color in [
        ("train_chosen_reward", "Chosen reward", REWARD_CHOSEN),
        ("train_rejected_reward", "Rejected reward", REWARD_REJECTED),
        ("train_reward_margin", "Reward margin", REWARD_MARGIN),
    ]:
        raw = df[col].values
        ax.plot(steps, raw, color=color, alpha=0.22, linewidth=0.7)
        if smoothing > 0:
            ax.plot(steps, _ema(raw, smoothing), color=color, linewidth=2, label=label)
        else:
            ax.plot(steps, raw, color=color, linewidth=1.5, label=label)

    ax.axhline(0, color=PALETTE["mid_grey"], linewidth=0.6, linestyle="--")
    ax.set_xlabel(
        "Training samples", fontsize=11, color=PALETTE["dark_grey"], fontfamily="Inter"
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
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to metrics.csv produced by a training run.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save plots (defaults to same directory as the CSV).",
    )
    parser.add_argument(
        "--algo",
        type=str,
        default="",
        help=(
            "Algorithm name used in titles and filenames (e.g. SFT, DPO, GRPO). "
            "Auto-detected from the CSV path or columns when not provided."
        ),
    )
    parser.add_argument(
        "--nll",
        action="store_true",
        help="Append '+ NLL' to the algorithm label (for DPO with NLL loss term).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="NLL alpha value to display in the plot title.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help="KL beta value to display in the plot title.",
    )
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.8,
        help="EMA smoothing factor (0 = none, ~1 = heavy). Default: 0.8",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively instead of only saving.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help=(
            "Filename prefix for saved PNGs (default: derived from --algo). "
            "Use for long titles, e.g. --basename sft_to_dpo_plus_nll."
        ),
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    algo = args.algo
    if args.nll:
        algo += " + NLL"
    algo_lower = algo.lower().replace(" + ", "_plus_").replace(" ", "_")
    file_prefix = args.basename if args.basename else algo_lower

    param_parts = []
    if args.beta is not None:
        param_parts.append(f"β={args.beta:g}")
    if args.alpha is not None:
        param_parts.append(f"α={args.alpha:g}")
    params = ", ".join(param_parts) or None

    has_reward_cols = {"train_chosen_reward", "train_rejected_reward"}.issubset(
        df.columns
    )

    plot_training_loss(
        df,
        algo=algo,
        params=params,
        smoothing=args.smoothing,
        save_path=str(output_dir / f"{file_prefix}_training_loss.png"),
    )

    if has_reward_cols:
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
