"""Plot training curves from LLM fine-tuning metrics CSVs.

Reads the ``metrics.csv`` produced by :func:`finetune_llm_sft` or
:func:`finetune_llm_preference` and produces publication-quality loss and
reward-margin plots.

Usage::

    python benchmarking/plot_llm_metrics.py saved_lora/20260402_093644/metrics.csv
    python benchmarking/plot_llm_metrics.py saved_lora/run1/metrics.csv -o docs/tutorials/llm_finetuning
    python benchmarking/plot_llm_metrics.py saved_lora/run1/metrics.csv --smoothing 0.85
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Colour palette – teal tones matching the AgileRL brand (#245c5c)
# ---------------------------------------------------------------------------
COLORS = {
    "loss": "#245c5c",
    "chosen": "#2a9d8f",
    "rejected": "#e76f51",
    "margin": "#264653",
}

# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------


def _ema(values: np.ndarray, alpha: float) -> np.ndarray:
    """Exponential moving average for smoothing noisy training curves."""
    smoothed = np.empty_like(values, dtype=float)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
    return smoothed


def _apply_style(ax: plt.Axes) -> None:
    """Apply a clean, modern style to an axes object."""
    ax.set_facecolor("#fafafa")
    ax.grid(True, color="#e0e0e0", linewidth=0.6, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#cccccc")
    ax.spines["bottom"].set_color("#cccccc")
    ax.tick_params(colors="#555555", labelsize=9)


def plot_dpo_training_loss(
    df: pd.DataFrame,
    *,
    smoothing: float = 0.8,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot the DPO training loss curve.

    :param df: DataFrame with at least ``step`` and ``train_loss`` columns.
    :param smoothing: EMA smoothing factor (0 = none, close to 1 = heavy).
    :param save_path: If provided, save the figure to this path.
    :return: The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    _apply_style(ax)

    steps = df["step"].values
    loss = df["train_loss"].values

    ax.plot(steps, loss, color=COLORS["loss"], alpha=0.25, linewidth=0.8)

    if smoothing > 0:
        # Smooth in log space so the EMA tracks the visual scale properly
        log_loss = np.log(np.clip(loss, 1e-12, None))
        smoothed = np.exp(_ema(log_loss, smoothing))
        ax.plot(
            steps,
            smoothed,
            color=COLORS["loss"],
            linewidth=2,
            label="Training loss (smoothed)",
        )
    else:
        ax.plot(
            steps,
            loss,
            color=COLORS["loss"],
            linewidth=1.5,
            label="Training loss",
        )

    ax.set_yscale("log")
    ax.set_xlabel("Training samples", fontsize=11, color="#333333")
    ax.set_ylabel("Loss", fontsize=11, color="#333333")
    ax.set_title("DPO Training Loss", fontsize=13, fontweight="bold", color="#222222")
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    return fig


def plot_dpo_reward_margins(
    df: pd.DataFrame,
    *,
    smoothing: float = 0.8,
    save_path: str | None = None,
) -> plt.Figure:
    """Plot chosen / rejected rewards and their margin over training.

    :param df: DataFrame with ``step``, ``train_chosen_reward``,
        ``train_rejected_reward``, and ``train_reward_margin`` columns.
    :param smoothing: EMA smoothing factor.
    :param save_path: If provided, save the figure to this path.
    :return: The matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    _apply_style(ax)

    steps = df["step"].values

    for col, label, color in [
        ("train_chosen_reward", "Chosen reward", COLORS["chosen"]),
        ("train_rejected_reward", "Rejected reward", COLORS["rejected"]),
        ("train_reward_margin", "Reward margin", COLORS["margin"]),
    ]:
        raw = df[col].values
        ax.plot(steps, raw, color=color, alpha=0.18, linewidth=0.7)
        if smoothing > 0:
            ax.plot(steps, _ema(raw, smoothing), color=color, linewidth=2, label=label)
        else:
            ax.plot(steps, raw, color=color, linewidth=1.5, label=label)

    ax.axhline(0, color="#999999", linewidth=0.6, linestyle="--")
    ax.set_xlabel("Training samples", fontsize=11, color="#333333")
    ax.set_ylabel("Reward", fontsize=11, color="#333333")
    ax.set_title(
        "DPO Reward Margins", fontsize=13, fontweight="bold", color="#222222"
    )
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        print(f"Saved: {save_path}")
    return fig


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


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
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    has_dpo_cols = {"train_chosen_reward", "train_rejected_reward"}.issubset(
        df.columns
    )

    plot_dpo_training_loss(
        df,
        smoothing=args.smoothing,
        save_path=str(output_dir / "dpo_training_loss.png"),
    )

    if has_dpo_cols:
        plot_dpo_reward_margins(
            df,
            smoothing=args.smoothing,
            save_path=str(output_dir / "dpo_reward_margins.png"),
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
