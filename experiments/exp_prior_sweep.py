"""Experiment: Prior-Strength Sweep.

Sweeps prior_strength from low to high and records:
  - acc(FROZEN)  — baseline with no training
  - acc(RANDOM)  — training with uninformative Bernoulli reward
  - acc(TRUE)    — training with verifier-correct reward

Derived quantities:
  - FE gain = acc(RANDOM) - acc(FROZEN)   [filtering/elicitation]
  - RD gain = acc(TRUE)   - acc(RANDOM)   [reward design]

Expected finding: as prior strength increases, FE gain rises and RD gain
shrinks — exactly the "Spurious Rewards" phenomenon, now explained as a
structural effect of filtering surfacing the latent prior.

Outputs
-------
results/prior_sweep.json
figures/fig_prior_sweep.{pdf,png}
"""

import sys
import os
import json
import argparse
import numpy as np

# Make src importable when called from experiments/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import plotstyle
plotstyle.apply()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.partition import prior_sweep

# Paper figures dir (relative to experiments/)
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--n_seeds",     type=int, default=8)
    parser.add_argument("--n_steps",     type=int, default=200)
    parser.add_argument("--n_prompts",   type=int, default=40)
    parser.add_argument("--n_actions",   type=int, default=8)
    parser.add_argument("--G",           type=int, default=8)
    parser.add_argument("--lr",          type=float, default=0.05)
    parser.add_argument("--beta",        type=float, default=0.01)
    parser.add_argument("--batch_size",  type=int, default=16)
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # -------------------------------------------------------------------
    prior_strengths = [0.20, 0.35, 0.50, 0.65, 0.80]

    print(f"Running prior sweep over {prior_strengths} ...")
    rows = prior_sweep(
        prior_strengths=prior_strengths,
        n_prompts=args.n_prompts,
        n_actions=args.n_actions,
        n_seeds=args.n_seeds,
        n_steps=args.n_steps,
        base_seed=args.seed,
        G=args.G,
        lr=args.lr,
        beta=args.beta,
        batch_size=args.batch_size,
    )

    # Save JSON
    out_path = os.path.join(RESULTS_DIR, "prior_sweep.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"  saved {out_path}")

    # -------------------------------------------------------------------
    # Plot
    ps_vals    = [r["prior_strength"] for r in rows]
    elicit_v   = [r["fe_gain"]        for r in rows]   # elicitation = acc(SPUR)-acc(RAND)
    rd_v       = [r["rd_gain"]        for r in rows]   # reward-design = acc(TRUE)-acc(SPUR)
    naive_v    = [r["naive_gain"]     for r in rows]   # acc(TRUE)-acc(RAND)  (conflated)
    rdfrac_v   = [r["rd_fraction"]    for r in rows]   # genuine RD share of naive
    nullv      = [r["random_null"]    for r in rows]   # acc(RAND)-acc(FROZEN) ~0
    frozen_v   = [r["acc_frozen"]     for r in rows]
    random_v   = [r["acc_random"]     for r in rows]
    spurious_v = [r["acc_spurious"]   for r in rows]
    true_v     = [r["acc_true"]       for r in rows]
    xs = np.array(ps_vals)

    def make_fig():
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2))

        # Panel 1: absolute accuracy curves for the 4 conditions
        ax = axes[0]
        ax.plot(ps_vals, frozen_v,   "o-", label="FROZEN (no RL)",       color=plotstyle.COLORS["gray"])
        ax.plot(ps_vals, random_v,   "s-", label="RANDOM (true-random)", color=plotstyle.COLORS["orange"])
        ax.plot(ps_vals, spurious_v, "^-", label="SPURIOUS (self-cons.)",color=plotstyle.COLORS["green"])
        ax.plot(ps_vals, true_v,     "D-", label="TRUE (verifiable)",    color=plotstyle.COLORS["blue"])
        ax.set_xlabel("Prior strength")
        ax.set_ylabel("Final accuracy (pass@1)")
        ax.set_title("(a) Accuracy of the four reward conditions")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)

        # Panel 2: exact telescoping decomposition (stacked) vs naive
        ax2 = axes[1]
        width = 0.035
        ax2.bar(xs - width/2, elicit_v, width=width,
                color=plotstyle.COLORS["green"], label="Elicitation (SPUR$-$RAND)")
        ax2.bar(xs + width/2, rd_v, width=width,
                color=plotstyle.COLORS["blue"], label="Reward design (TRUE$-$SPUR)")
        ax2.plot(xs, naive_v, "k--o", label="Naive 'TRUE$-$RANDOM'", linewidth=1.6, markersize=4)
        ax2.axhline(0, color="k", linewidth=0.8)
        ax2.set_xlabel("Prior strength")
        ax2.set_ylabel("Accuracy gain")
        ax2.set_title("(b) Exact partition vs naive estimand")
        ax2.legend(fontsize=8)

        # Panel 3: the headline — RD fraction of the naive estimand collapses
        ax3 = axes[2]
        ax3.plot(ps_vals, rdfrac_v, "o-", color=plotstyle.COLORS["red"], linewidth=2.2)
        ax3.axhline(1.0, color="k", linewidth=0.8, linestyle=":")
        ax3.axhline(0.0, color="k", linewidth=0.8)
        ax3.fill_between(ps_vals, rdfrac_v, 1.0, alpha=0.12, color=plotstyle.COLORS["green"])
        ax3.set_xlabel("Prior strength")
        ax3.set_ylabel("Reward-design fraction of 'TRUE$-$RANDOM'")
        ax3.set_title("(c) How much of 'true$-$random' is really reward design?")
        ax3.annotate("conflation grows\nwith prior strength",
                     xy=(ps_vals[-1], rdfrac_v[-1]),
                     xytext=(ps_vals[1], 0.35), fontsize=8, color="gray",
                     arrowprops=dict(arrowstyle="->", color="gray"))

        fig.suptitle(
            "What does 'true$-$random' estimate? Elicitation (self-consistency) vs genuine reward design",
            fontsize=12, y=1.02)
        fig.tight_layout()
        return fig

    base_path = os.path.join(FIGURES_DIR, "fig_prior_sweep")
    plotstyle.save_fig(make_fig(), base_path)
    local_figs = os.path.join(RESULTS_DIR, "figures")
    os.makedirs(local_figs, exist_ok=True)
    plotstyle.save_fig(make_fig(), os.path.join(local_figs, "fig_prior_sweep"))

    # Print summary
    print("\nPrior sweep results (exact telescoping decomposition):")
    print(f"{'ps':>5}  {'froz':>6}  {'rand':>6}  {'spur':>6}  {'true':>6}  "
          f"{'null':>6}  {'elicit':>7}  {'rwd-des':>7}  {'naive':>6}  {'RDfrac':>6}")
    for r in rows:
        print(f"{r['prior_strength']:>5.2f}  {r['acc_frozen']:>6.3f}  "
              f"{r['acc_random']:>6.3f}  {r['acc_spurious']:>6.3f}  {r['acc_true']:>6.3f}  "
              f"{r['random_null']:>+6.3f}  {r['fe_gain']:>+7.3f}  {r['rd_gain']:>+7.3f}  "
              f"{r['naive_gain']:>+6.3f}  {r['rd_fraction']:>6.2f}")


if __name__ == "__main__":
    main()
