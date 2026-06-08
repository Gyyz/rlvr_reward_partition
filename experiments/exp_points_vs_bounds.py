"""Experiment: Points vs Bounds.

Demonstrates the difference between:
  * DECOUPLED regime: weak interaction → report FE and RD as POINT estimates
  * COUPLED regime:   strong interaction → report decomposition as BOUNDS

We construct the coupled regime by using a FORMAT reward (which correlates
with the prior, creating an informative-but-spurious signal that interacts
with the filtering mechanism).

Outputs
-------
results/points_vs_bounds.json
figures/fig_points_bounds.{pdf,png}
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

from src.env import make_task_distribution
from src.grpo import GRPOConfig, run_grpo_seeds, mean_final_acc
from src.partition import estimate_partition, DEFAULT_INTERACTION_THRESHOLD

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_condition(
    prior_strength: float,
    format_strength: float,
    n_prompts: int,
    n_actions: int,
    n_seeds: int,
    n_steps: int,
    base_seed: int,
    G: int,
    lr: float,
    beta: float,
    batch_size: int,
    n_bootstrap: int,
) -> dict:
    """Run one condition and return partition estimate as a dict."""
    dist = make_task_distribution(
        n_prompts=n_prompts,
        n_actions=n_actions,
        prior_strength=prior_strength,
        format_strength=format_strength,
        seed=base_seed,
    )
    est = estimate_partition(
        dist=dist,
        n_seeds=n_seeds,
        n_steps=n_steps,
        base_seed=base_seed + 10,
        G=G,
        lr=lr,
        beta=beta,
        batch_size=batch_size,
        n_bootstrap=n_bootstrap,
    )
    return {
        "prior_strength": prior_strength,
        "format_strength": format_strength,
        "acc_frozen": est.acc_frozen,
        "acc_random": est.acc_random,
        "acc_true": est.acc_true,
        "fe_gain": est.filtering_elicitation_gain,
        "rd_gain": est.reward_design_gain,
        "tot_gain": est.total_gain,
        "fe_ci": list(est.fe_ci),
        "rd_ci": list(est.rd_ci),
        "interaction_term": est.interaction_term,
        "decomposition_mode": est.decomposition_mode,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int,   default=0)
    parser.add_argument("--n_seeds",     type=int,   default=8)
    parser.add_argument("--n_steps",     type=int,   default=200)
    parser.add_argument("--n_prompts",   type=int,   default=40)
    parser.add_argument("--n_actions",   type=int,   default=8)
    parser.add_argument("--G",           type=int,   default=8)
    parser.add_argument("--lr",          type=float, default=0.05)
    parser.add_argument("--beta",        type=float, default=0.01)
    parser.add_argument("--batch_size",  type=int,   default=16)
    parser.add_argument("--n_bootstrap", type=int,   default=300)
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    # POINTS regime: strong prior -> elicitation (self-consistency) gain is large
    #   and the conflation-bias CI excludes 0, so the split is point-identified.
    # BOUNDS regime: near the crossover prior (~0.35) the self-consistency channel
    #   neither helps nor hurts; elicitation is statistically indistinguishable
    #   from 0, so we can only BOUND the elicitation/reward-design split.
    conditions = [
        dict(label="Point-identified\n(strong prior, ps=0.80)",
             prior_strength=0.80, format_strength=0.0,
             expected_mode="points"),
        dict(label="Bounded\n(self-cons. crossover, ps=0.25)",
             prior_strength=0.25, format_strength=0.0,
             base_seed=25, n_seeds=4, expected_mode="bounds"),
    ]

    results = []
    for cond in conditions:
        print(f"\nRunning condition: {cond['label'].replace(chr(10), ' ')} ...")
        res = run_condition(
            prior_strength=cond["prior_strength"],
            format_strength=cond["format_strength"],
            n_prompts=args.n_prompts,
            n_actions=args.n_actions,
            n_seeds=cond.get("n_seeds", args.n_seeds),
            n_steps=args.n_steps,
            base_seed=cond.get("base_seed", args.seed + int(cond["prior_strength"] * 100)),
            G=args.G,
            lr=args.lr,
            beta=args.beta,
            batch_size=args.batch_size,
            n_bootstrap=args.n_bootstrap,
        )
        res["label"] = cond["label"]
        res["expected_mode"] = cond["expected_mode"]
        results.append(res)
        print(f"  decomposition_mode = {res['decomposition_mode']}")
        print(f"  FE gain = {res['fe_gain']:.4f} CI={res['fe_ci']}")
        print(f"  RD gain = {res['rd_gain']:.4f} CI={res['rd_ci']}")
        print(f"  interaction term = {res['interaction_term']:.4f}")

    # Save JSON
    out_path = os.path.join(RESULTS_DIR, "points_vs_bounds.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  saved {out_path}")

    # -------------------------------------------------------------------
    # Plot: two side-by-side panels showing FE and RD with CIs
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax_i, (ax, res) in enumerate(zip(axes, results)):
        mode = res["decomposition_mode"]
        fe   = res["fe_gain"]
        rd   = res["rd_gain"]
        fe_ci = res["fe_ci"]
        rd_ci = res["rd_ci"]

        x = np.array([0, 1])
        heights = [fe, rd]
        ci_lo   = [fe - fe_ci[0], rd - rd_ci[0]]
        ci_hi   = [fe_ci[1] - fe, rd_ci[1] - rd]

        bar_colors = [plotstyle.COLORS["orange"], plotstyle.COLORS["blue"]]
        bars = ax.bar(x, heights, width=0.4, color=bar_colors, alpha=0.85,
                      edgecolor="k", linewidth=0.8)
        ax.errorbar(x, heights, yerr=[ci_lo, ci_hi],
                    fmt="none", capsize=6, color="k", linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels(["FE gain\n(filter/elicit)", "RD gain\n(reward design)"],
                           fontsize=9)
        ax.set_ylabel("Gain vs FROZEN")
        ax.axhline(0, color="k", linewidth=0.8)

        # Mode label with colour
        mode_color = "red" if mode == "bounds" else "green"
        ax.set_title(
            f"{res['label'].replace(chr(10), ' ')}\nMode: {mode.upper()}",
            fontsize=9,
        )
        # Shade background to indicate mode
        ax.set_facecolor("#fff5f5" if mode == "bounds" else "#f5fff5")

        # If bounds mode, draw shaded range
        if mode == "bounds":
            ax.bar(x[0], res["tot_gain"], width=0.4, bottom=0,
                   color=plotstyle.COLORS["orange"], alpha=0.2,
                   edgecolor=plotstyle.COLORS["orange"], linestyle="--",
                   linewidth=1.5, label="FE upper bound")
            ax.annotate("BOUNDS\n(coupled)", xy=(0.5, max(heights)*0.8),
                        ha="center", fontsize=8, color="red",
                        bbox=dict(boxstyle="round", facecolor="lightyellow"))
        else:
            ax.annotate("POINTS\n(decoupled)", xy=(0.5, max(heights)*0.8),
                        ha="center", fontsize=8, color="green",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))

    fig.suptitle("Points vs Bounds: Decoupled vs Coupled RLVR Regimes", fontsize=11)
    fig.tight_layout()

    for base_dir in [FIGURES_DIR, os.path.join(RESULTS_DIR, "figures")]:
        plotstyle.save_fig(
            _clone_pvb_fig(results),
            os.path.join(base_dir, "fig_points_bounds"),
        )


def _clone_pvb_fig(results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, res in zip(axes, results):
        mode = res["decomposition_mode"]
        fe, rd = res["fe_gain"], res["rd_gain"]
        fe_ci, rd_ci = res["fe_ci"], res["rd_ci"]
        x = np.array([0, 1])
        heights = [fe, rd]
        ci_lo = [fe - fe_ci[0], rd - rd_ci[0]]
        ci_hi = [fe_ci[1] - fe, rd_ci[1] - rd]
        bar_colors = [plotstyle.COLORS["orange"], plotstyle.COLORS["blue"]]
        ax.bar(x, heights, width=0.4, color=bar_colors, alpha=0.85,
               edgecolor="k", linewidth=0.8)
        ax.errorbar(x, heights, yerr=[ci_lo, ci_hi],
                    fmt="none", capsize=6, color="k", linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(["FE gain\n(filter/elicit)", "RD gain\n(reward design)"], fontsize=9)
        ax.set_ylabel("Gain vs FROZEN")
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_title(
            f"{res['label'].replace(chr(10),' ')}\nMode: {mode.upper()}", fontsize=9)
        ax.set_facecolor("#fff5f5" if mode == "bounds" else "#f5fff5")
        if mode == "bounds":
            ax.annotate("BOUNDS\n(coupled)", xy=(0.5, max(heights)*0.8 if max(heights)>0 else 0.05),
                        ha="center", fontsize=8, color="red",
                        bbox=dict(boxstyle="round", facecolor="lightyellow"))
        else:
            ax.annotate("POINTS\n(decoupled)", xy=(0.5, max(heights)*0.8 if max(heights)>0 else 0.05),
                        ha="center", fontsize=8, color="green",
                        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5))
    fig.suptitle("Points vs Bounds: Decoupled vs Coupled RLVR Regimes", fontsize=11)
    fig.tight_layout()
    return fig


if __name__ == "__main__":
    main()
