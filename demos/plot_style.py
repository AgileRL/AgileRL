"""Reusable matplotlib styling for AgileRL figures.

Provides the design-token palette, semantic series colours (brand teal primary,
light mint, orange accent), axis/figure helpers, and Inter font registration.
Inter TTFs are loaded from the user cache (``~/.cache/agilerl/fonts/inter`` on
Linux/macOS) and downloaded from the pinned Inter release if missing.
"""

from __future__ import annotations

import io
import os
import urllib.request
import warnings
import zipfile
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib import font_manager

if TYPE_CHECKING:
    from matplotlib.axes import Axes

# ---------------------------------------------------------------------------
# Design tokens (Figma / product palette)
# ---------------------------------------------------------------------------
PALETTE: dict[str, str] = {
    "very_light_grey": "#ededef",
    "off_black": "#0A0A0C",
    "dark_grey": "#6C6C89",
    "alert_green": "#0A9467",
    "alert_red": "#7A2626",
    "mid_grey": "#D1D1DB",
    "brand": "#467F81",
    "brand_50": "#A3BFC0",
    "brand_20": "#CCD7D9",
    "brand_5": "#F6F9F9",
    "off_white": "#FAFAFA",
    "gold": "#F6D173",
    "required_red": "#F53D6B",
    "white": "#FFFFFF",
    "warning_orange": "#F57F12",
    "error_red": "#ED4A4A",
    "blue": "#519DE9",
}

# Series semantics: brand teal = primary; lighter brand_50 + orange for contrast
PRIMARY = PALETTE["brand"]
LIGHT_GREEN = PALETTE["brand_50"]
ACCENT_ORANGE = PALETTE["warning_orange"]

INTER_ZIP_URL = "https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip"
INTER_FONT_FILES = ("Inter-Regular.ttf", "Inter-SemiBold.ttf", "Inter-Bold.ttf")


def inter_font_cache_dir() -> Path:
    """Directory where Inter TTFs are stored (created on demand)."""
    if xdg := os.environ.get("XDG_CACHE_HOME"):
        base = Path(xdg)
    else:
        base = Path.home() / ".cache"
    return base / "agilerl" / "fonts" / "inter"


def ensure_inter_fonts() -> Path:
    """Ensure Inter Regular / SemiBold / Bold exist locally; download zip if needed.

    :returns: Path to the directory containing the ``.ttf`` files.
    :raises OSError: If extraction or download fails (e.g. offline with empty cache).
    """
    dest = inter_font_cache_dir()
    dest.mkdir(parents=True, exist_ok=True)
    if all((dest / f).is_file() for f in INTER_FONT_FILES):
        return dest

    with urllib.request.urlopen(INTER_ZIP_URL, timeout=120) as response:
        data = response.read()

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for name in INTER_FONT_FILES:
            member = f"extras/ttf/{name}"
            with zf.open(member) as src, (dest / name).open("wb") as out:
                out.write(src.read())
    return dest


@lru_cache(maxsize=1)
def register_inter_font() -> None:
    """Register Inter from cache and set matplotlib defaults (idempotent)."""
    try:
        font_dir = ensure_inter_fonts()
    except OSError as exc:
        warnings.warn(
            f"Inter fonts unavailable ({exc!s}); using fallback sans-serif.",
            stacklevel=2,
        )
        font_dir = None

    if font_dir is not None:
        for name in INTER_FONT_FILES:
            path = font_dir / name
            if path.is_file():
                font_manager.fontManager.addfont(str(path))

    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [
        "Inter",
        "DejaVu Sans",
        "Helvetica",
        "Arial",
        "sans-serif",
    ]
    plt.rcParams["axes.labelcolor"] = PALETTE["dark_grey"]
    plt.rcParams["xtick.color"] = PALETTE["dark_grey"]
    plt.rcParams["ytick.color"] = PALETTE["dark_grey"]
    plt.rcParams["axes.edgecolor"] = PALETTE["mid_grey"]
    plt.rcParams["grid.color"] = PALETTE["mid_grey"]
    plt.rcParams["grid.linewidth"] = 0.6
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["figure.facecolor"] = PALETTE["white"]
    plt.rcParams["axes.facecolor"] = PALETTE["off_white"]


def style_axes(ax: Axes) -> None:
    """Apply AgileRL chart chrome (grid, spines, background)."""
    register_inter_font()
    ax.set_facecolor(PALETTE["off_white"])
    ax.grid(True, color=PALETTE["mid_grey"], linewidth=0.6, linestyle="-")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(PALETTE["mid_grey"])
    ax.spines["bottom"].set_color(PALETTE["mid_grey"])
    ax.tick_params(colors=PALETTE["dark_grey"], labelsize=9)


def legend_kw() -> dict[str, object]:
    """Keyword arguments for a consistent legend box."""
    return {
        "frameon": True,
        "fancybox": True,
        "framealpha": 0.95,
        "fontsize": 9,
        "edgecolor": PALETTE["mid_grey"],
        "facecolor": PALETTE["brand_5"],
    }
