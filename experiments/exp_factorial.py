"""Experiment: 2×2×2 Factorial (reward × filtering × prior strength).

Runs 8 cells (2^3), estimates main effects + interactions via a linear
model (effect coding), and applies the additivity specification test to
determine whether the FE/RD decomposition is reported as POINTS or BOUNDS.

Outputs
-------
results/factorial.json  — full 8-cell table + linear model coefficients
figures/fig_factorial.{pdf,png}  — bar chart of cell means + effects
"""

import sys
import os
import json
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import plotstyle
plotstyle.apply()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.partition import run_factorial, factorial_to_dict

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--n_seeds",     type=int,   default=8)
    parser.add_argument("--n_steps",     type=int,   default=200)
    parser.add_argument("--n_prompts",   type=int,   default=40)
    parser.add_argument("--n_actions",   type=int,   default=8)
    parser.add_argument("--hi_prior",    type=float, default=0.80)
    parser.add_argument("--lo_prior",    type=float, default=0.35)
    parser.add_argument("--G",           type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=0.05)
    parser.add_argument("--beta",        type=float, default=0.01)
    parser.add_argument("--batch_size",  type=int,   default=16)
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    print("Running 2×2×2 factorial ...")
    fr = run_factorial(
        n_prompts=args.n_prompts,
        n_actions=args.n_actions,
        n_seeds=args.n_seeds,
        n_steps=args.n_steps,
        hi_prior=args.hi_prior,
        lo_prior=args.lo_prior,
        G=args.G,
        lr=args.lr,
        beta=args.beta,
        batch_size=args.batch_size,
        base_seed=args.seed,
    )

    # Save JSON
    out = factorial_to_dict(fr)
    out_path = os.path.join(RESULTS_DIR, "factorial.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved {out_path}")

    # Print summary
    print(f"\n  Additivity test: {fr.additivity_test}")
    print(f"  Interaction ratio: {fr.interaction_ratio:.4f}")
    print(f"  Decomposition mode: {fr.decomposition_mode}")
    print("\nLinear model effects:")
    for name, val in fr.effects.items():
        print(f"  {name:15s}: {val:+.4f}")

    print("\n8-cell table:")
    hdr = f"{'reward':>6}  {'filter':>6}  {'prior':>6}  {'mean_acc':>9}  {'std_acc':>8}"
    print(hdr)
    for c in fr.cells:
        rstr = "TRUE" if c.reward_informative else "SELFC"
        fstr = "on"   if c.filtering          else "off"
        pstr = f"{c.prior_strength:.2f}"
        print(f"  {rstr:>6}  {fstr:>6}  {pstr:>6}  {c.mean_acc:>9.4f}  {c.std_acc:>8.4f}")

    # -------------------------------------------------------------------
    # Plot: bar chart of 8 cells, grouped by prior strength
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

    for ax_i, ps_level in enumerate(["hi", "lo"]):
        ax = axes[ax_i]
        ps_val = fr.hi_prior if ps_level == "hi" else fr.lo_prior
        cells_ps = [c for c in fr.cells if abs(c.prior_strength - ps_val) < 1e-6]

        labels = []
        heights = []
        errors = []
        colors = []

        for c in cells_ps:
            rstr = "TRUE" if c.reward_informative else "SELFC"
            fstr = "filt-on" if c.filtering else "filt-off"
            labels.append(f"{rstr}\n{fstr}")
            heights.append(c.mean_acc)
            errors.append(c.std_acc)
            colors.append(plotstyle.COLORS["blue"] if c.reward_informative
                          else plotstyle.COLORS["orange"])

        x = np.arange(len(labels))
        bars = ax.bar(x, heights, yerr=errors, capsize=4, width=0.55,
                      color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"Prior strength = {'HIGH' if ps_level=='hi' else 'LOW'} ({ps_val:.2f})")
        ax.set_ylabel("Final accuracy (pass@1)")
        ax.set_ylim(0, 1)

        # Add text labels
        for bar, h, e in zip(bars, heights, errors):
            ax.text(bar.get_x() + bar.get_width()/2, h + e + 0.01,
                    f"{h:.3f}", ha="center", va="bottom", fontsize=7)

    # Legend
    blue_patch   = mpatches.Patch(color=plotstyle.COLORS["blue"],   label="TRUE reward")
    orange_patch = mpatches.Patch(color=plotstyle.COLORS["orange"], label="SELF-CONS (spurious) reward")
    axes[0].legend(handles=[blue_patch, orange_patch], loc="upper left", fontsize=9)

    # Additivity note
    note = (f"Additivity test: {fr.additivity_test} "
            f"(interaction ratio={fr.interaction_ratio:.3f})\n"
            f"Decomposition mode: {fr.decomposition_mode.upper()}")
    fig.text(0.5, -0.04, note, ha="center", fontsize=9,
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # Effects bar (inset or separate panel)
    fig.suptitle("2x2x2 Factorial: Reward (TRUE vs SELF-CONS) x Filtering x Prior", fontsize=12)
    fig.tight_layout()

    for base_dir in [FIGURES_DIR, os.path.join(RESULTS_DIR, "figures")]:
        plotstyle.save_fig(
            fig if base_dir == FIGURES_DIR else _clone_fig(fr),
            os.path.join(base_dir, "fig_factorial"),
        )

    # Second figure: linear model main effects
    _plot_effects(fr, FIGURES_DIR, RESULTS_DIR)


def _clone_fig(fr):
    """Re-create the factorial figure for a second save."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax_i, ps_level in enumerate(["hi", "lo"]):
        ax = axes[ax_i]
        ps_val = fr.hi_prior if ps_level == "hi" else fr.lo_prior
        cells_ps = [c for c in fr.cells if abs(c.prior_strength - ps_val) < 1e-6]
        labels, heights, errors, colors = [], [], [], []
        for c in cells_ps:
            rstr = "TRUE" if c.reward_informative else "SELFC"
            fstr = "filt-on" if c.filtering else "filt-off"
            labels.append(f"{rstr}\n{fstr}")
            heights.append(c.mean_acc)
            errors.append(c.std_acc)
            colors.append(plotstyle.COLORS["blue"] if c.reward_informative
                          else plotstyle.COLORS["orange"])
        x = np.arange(len(labels))
        ax.bar(x, heights, yerr=errors, capsize=4, width=0.55,
               color=colors, alpha=0.85, edgecolor="k", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(f"Prior = {'HIGH' if ps_level=='hi' else 'LOW'} ({ps_val:.2f})")
        ax.set_ylabel("Final accuracy (pass@1)")
        ax.set_ylim(0, 1)
    fig.suptitle("2x2x2 Factorial: Reward (TRUE vs SELF-CONS) x Filtering x Prior", fontsize=12)
    fig.tight_layout()
    return fig


def _plot_effects(fr, figures_dir, results_dir):
    """Plot linear model main effects + interactions."""
    effects = fr.effects
    names   = list(effects.keys())[1:]  # skip intercept
    values  = [effects[n] for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [
        plotstyle.COLORS["blue"]   if abs(v) == max(abs(x) for x in values[:3]) and names[i] == "A_reward"
        else plotstyle.COLORS["orange"] if i < 3
        else plotstyle.COLORS["purple"]
        for i, v in enumerate(values)
    ]
    # Simpler: main effects in blue family, interactions in gray
    bar_colors = (
        [plotstyle.COLORS["blue"], plotstyle.COLORS["orange"], plotstyle.COLORS["green"]]
        + [plotstyle.COLORS["gray"]] * 4
    )
    bars = ax.bar(np.arange(len(names)), values, color=bar_colors, edgecolor="k", linewidth=0.5)
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(["A: reward", "B: filter", "C: prior",
                         "AB", "AC", "BC", "ABC"], rotation=20, ha="right")
    ax.set_ylabel("Effect size (accuracy)")
    ax.set_title(f"Linear model effects\n(additivity: {fr.additivity_test}, "
                 f"interaction ratio={fr.interaction_ratio:.3f})")
    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                v + np.sign(v) * 0.002,
                f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig.tight_layout()

    for base_dir in [figures_dir, os.path.join(results_dir, "figures")]:
        _save_eff_fig(fr, base_dir)


def _save_eff_fig(fr, base_dir):
    effects = fr.effects
    names   = list(effects.keys())[1:]
    values  = [effects[n] for n in names]
    bar_colors = (
        [plotstyle.COLORS["blue"], plotstyle.COLORS["orange"], plotstyle.COLORS["green"]]
        + [plotstyle.COLORS["gray"]] * 4
    )
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    bars = ax2.bar(np.arange(len(names)), values, color=bar_colors, edgecolor="k", linewidth=0.5)
    ax2.axhline(0, color="k", linewidth=0.8)
    ax2.set_xticks(np.arange(len(names)))
    ax2.set_xticklabels(["A: reward", "B: filter", "C: prior",
                          "AB", "AC", "BC", "ABC"], rotation=20, ha="right")
    ax2.set_ylabel("Effect size (accuracy)")
    ax2.set_title(f"Linear model effects\n(additivity: {fr.additivity_test}, "
                  f"interaction ratio={fr.interaction_ratio:.3f})")
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 v + np.sign(v) * 0.002,
                 f"{v:+.4f}", ha="center", va="bottom" if v >= 0 else "top", fontsize=7)
    fig2.tight_layout()
    plotstyle.save_fig(fig2, os.path.join(base_dir, "fig_factorial_effects"))


if __name__ == "__main__":
    main()
