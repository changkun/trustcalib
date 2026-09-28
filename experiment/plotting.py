"""Shared figure style for the manuscript figures.

Figures are drawn at their final printed size (the manuscript's text width is
6.5 in), so the font sizes below are the sizes on the page. The body font is
Computer Modern, matching the manuscript. One colour per concept, used
identically in every figure.
"""

from __future__ import annotations

import logging

import matplotlib as mpl

TEXT_WIDTH = 6.5  # inches

COLORS = {
    # models
    "product": "#6B7280",
    "additive": "#0F766E",
    "linear": "#7C3AED",
    "per-tool": "#B45309",
    "per-cell": "#2563EB",
    "oracle": "#111111",
    # decisions (Okabe-Ito)
    "allow": "#009E73",
    "ask": "#E69F00",
    "block": "#D55E00",
    # judge error rates
    "fa": "#D55E00",
    "fb": "#0072B2",
    # annotations
    "marker": "#374151",
}

RC = {
    "font.family": "serif",
    "font.serif": ["cmr10"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": True,
    "font.size": 8,
    "axes.titlesize": 8,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "legend.borderaxespad": 0.3,
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 4,
    "axes.labelpad": 2.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.major.pad": 2,
    "ytick.major.pad": 2,
    "lines.linewidth": 1.3,
    "lines.markersize": 3.5,
    "patch.linewidth": 0.0,
    "hatch.linewidth": 0.6,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply() -> None:
    """Install the manuscript style for all subsequent figures."""
    mpl.rcParams.update(RC)
    # the bundled Computer Modern TTFs carry zero timestamps; embedding them warns
    logging.getLogger("fontTools").setLevel(logging.ERROR)


def panel(ax, letter: str, title: str = "") -> None:
    """Left-aligned panel title with a bold panel letter, e.g. "(a) Product kernel"."""
    ax.set_title(rf"$\mathbf{{({letter})}}$  {title}", loc="left")


def save(fig, path: str) -> None:
    fig.savefig(path)
    import matplotlib.pyplot as plt

    plt.close(fig)
