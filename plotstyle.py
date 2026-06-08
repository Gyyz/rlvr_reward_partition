"""Shared matplotlib styling for all paper figures.

Produces clean, conference-ready vector figures. Import and call `apply()`
at the top of any plotting script, then use the COLORS / save_fig helpers.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {
    "blue":   "#1f77b4",
    "red":    "#d62728",
    "green":  "#2ca02c",
    "orange": "#ff7f0e",
    "purple": "#9467bd",
    "gray":   "#7f7f7f",
    "brown":  "#8c564b",
    "teal":   "#17becf",
}
CYCLE = [COLORS[k] for k in ("blue", "red", "green", "orange", "purple", "teal", "brown", "gray")]


def apply():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.prop_cycle": plt.cycler(color=CYCLE),
    })


def save_fig(fig, path):
    """Save as both PDF (for LaTeX) and PNG (for preview)."""
    fig.savefig(path + ".pdf")
    fig.savefig(path + ".png")
    plt.close(fig)
    print(f"  wrote {path}.pdf / .png")
