"""Plot training curves from LLM fine-tuning metrics CSVs.

Reads the ``metrics.csv`` produced by :func:`finetune_llm_sft` or
:func:`finetune_llm_preference` and produces publication-quality loss and
reward-margin plots.

Usage::

    python benchmarking/plot_llm_metrics.py saved_lora/20260402_093644/metrics.csv
    python benchmarking/plot_llm_metrics.py outputs/20260402_SFT/metrics.csv -o docs/tutorials/llm_finetuning/images
    python benchmarking/plot_llm_metrics.py saved_lora/run1/metrics.csv --smoothing 0.85
    python benchmarking/plot_llm_metrics.py saved_lora/run1/metrics.csv --algo DPO
"""

from __future__ import annotations

import argparse
import os
import re
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


def _detect_algo(csv_path: Path, df: pd.DataFrame) -> str:
    """Best-effort algorithm detection from path or CSV columns.

    Checks the path components for known algorithm names (SFT, DPO, GRPO),
    then falls back to column heuristics.
    """
    path_str = str(csv_path).upper()
    for algo in ("SFT", "DPO", "GRPO"):
        if re.search(rf"[\b_/]{algo}[\b_/.]", path_str) or path_str.endswith(algo):
            return algo

    has_rewards = {"train_chosen_reward", "train_rejected_reward"}.issubset(df.columns)
    if has_rewards:
        return "DPO"
    return "SFT"


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
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    _apply_style(ax)

    steps = df["step"].values
    loss = df["train_loss"].values

    ax.plot(steps, loss, color=COLORS["loss"], alpha=0.25, linewidth=0.8)

    if smoothing > 0:
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
    ax.set_title(
        _build_title(algo, "Training Loss", params),
        fontsize=13,
        fontweight="bold",
        color="#222222",
    )
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
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
        _build_title(algo, "Reward Margins", params),
        fontsize=13,
        fontweight="bold",
        color="#222222",
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
        "--algo",
        type=str,
        default=None,
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
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)

    algo = args.algo or _detect_algo(csv_path, df)
    if args.nll:
        algo += " + NLL"
    algo_lower = algo.lower().replace(" + ", "_plus_").replace(" ", "_")

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
        save_path=str(output_dir / f"{algo_lower}_training_loss.png"),
    )

    if has_reward_cols:
        plot_reward_margins(
            df,
            algo=algo,
            params=params,
            smoothing=args.smoothing,
            save_path=str(output_dir / f"{algo_lower}_reward_margins.png"),
        )

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
