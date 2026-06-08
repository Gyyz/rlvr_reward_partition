"""Experiment: Power analysis / seed-to-seed variance.

Examines how the RD gain CI width varies with the number of seeds, and
whether the CI can exclude the pre-registered invalidation threshold
(DEFAULT_INTERACTION_THRESHOLD = 0.20 on the logit scale, translated
to accuracy space as 0.05 absolute accuracy point).

We define the INVALIDATION THRESHOLD as: if the RD gain CI includes 0
at the chosen number of seeds, the experiment is underpowered.

Outputs
-------
results/power.json
figures/fig_power.{pdf,png}
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

from src.env import make_task_distribution
from src.grpo import GRPOConfig, run_grpo_seeds
from src.partition import DEFAULT_INTERACTION_THRESHOLD

FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

INVALIDATION_THRESHOLD = 0.03  # 3 pp absolute RD gain — if CI includes this, underpowered


def estimate_rd_ci(
    dist,
    n_seeds: int,
    n_steps: int,
    base_seed: int,
    G: int,
    lr: float,
    beta: float,
    batch_size: int,
    n_bootstrap: int,
) -> tuple[float, float, float]:
    """Return (rd_mean, ci_lo, ci_hi)."""
    rng_bs = np.random.default_rng(base_seed + 77777)

    accs_random = []   # SPURIOUS (self-consistency) anchor — the genuine RD baseline
    accs_true   = []

    for reward_type, container in [("spurious", accs_random), ("true", accs_true)]:
        cfg = GRPOConfig(
            n_steps=n_steps,
            batch_size=batch_size,
            G=G,
            lr=lr,
            beta=beta,
            filtering=True,
            reward_type=reward_type,
            eval_every=50,
            eval_samples=64,
            seed=base_seed,
        )
        results = run_grpo_seeds(dist, cfg, n_seeds=n_seeds,
                                  base_seed=base_seed + (0 if reward_type=="spurious" else 300))
        container.extend([r.final_acc for r in results])

    arr_rand = np.array(accs_random)
    arr_true = np.array(accs_true)
    rd_mean  = float(arr_true.mean() - arr_rand.mean())

    # Bootstrap
    boots = []
    for _ in range(n_bootstrap):
        b_rand = rng_bs.choice(arr_rand, size=n_seeds, replace=True).mean()
        b_true = rng_bs.choice(arr_true, size=n_seeds, replace=True).mean()
        boots.append(b_true - b_rand)
    ci = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5)))
    return rd_mean, ci[0], ci[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",         type=int,   default=0)
    parser.add_argument("--n_steps",      type=int,   default=200)
    parser.add_argument("--n_prompts",    type=int,   default=40)
    parser.add_argument("--n_actions",    type=int,   default=8)
    parser.add_argument("--G",            type=int,   default=8)
    parser.add_argument("--lr",           type=float, default=0.05)
    parser.add_argument("--beta",         type=float, default=0.01)
    parser.add_argument("--batch_size",   type=int,   default=16)
    parser.add_argument("--n_bootstrap",  type=int,   default=300)
    parser.add_argument("--prior_strength", type=float, default=0.50)
    args = parser.parse_args()

    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    dist = make_task_distribution(
        n_prompts=args.n_prompts,
        n_actions=args.n_actions,
        prior_strength=args.prior_strength,
        seed=args.seed,
    )

    seed_counts = [4, 6, 8, 10, 12, 16]
    rows = []
    for n_seeds in seed_counts:
        print(f"  n_seeds={n_seeds} ...", end=" ", flush=True)
        rd_mean, ci_lo, ci_hi = estimate_rd_ci(
            dist=dist,
            n_seeds=n_seeds,
            n_steps=args.n_steps,
            base_seed=args.seed,
            G=args.G,
            lr=args.lr,
            beta=args.beta,
            batch_size=args.batch_size,
            n_bootstrap=args.n_bootstrap,
        )
        ci_width = ci_hi - ci_lo
        excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        excludes_threshold = (ci_hi < INVALIDATION_THRESHOLD) or (ci_lo > INVALIDATION_THRESHOLD)
        print(f"RD={rd_mean:.4f} CI=[{ci_lo:.4f},{ci_hi:.4f}]  "
              f"excl_zero={excludes_zero}  excl_thresh={excludes_threshold}")
        rows.append({
            "n_seeds": n_seeds,
            "rd_mean": rd_mean,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "ci_width": ci_width,
            "excludes_zero": excludes_zero,
            "excludes_invalidation_threshold": excludes_threshold,
        })

    out = {
        "invalidation_threshold": INVALIDATION_THRESHOLD,
        "prior_strength": args.prior_strength,
        "rows": rows,
    }
    out_path = os.path.join(RESULTS_DIR, "power.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  saved {out_path}")

    # -------------------------------------------------------------------
    # Plot
    def make_fig():
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: RD gain with CI vs n_seeds
        ax = axes[0]
        ns = [r["n_seeds"] for r in rows]
        rd = [r["rd_mean"] for r in rows]
        lo = [r["ci_lo"]   for r in rows]
        hi = [r["ci_hi"]   for r in rows]
        ax.plot(ns, rd, "o-", color=plotstyle.COLORS["blue"], label="RD gain (mean)")
        ax.fill_between(ns, lo, hi, alpha=0.25, color=plotstyle.COLORS["blue"], label="95% CI")
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
        ax.axhline(INVALIDATION_THRESHOLD, color=plotstyle.COLORS["red"],
                   linewidth=1.5, linestyle=":", label=f"Invalidation threshold ({INVALIDATION_THRESHOLD:.2f})")
        ax.set_xlabel("Number of seeds")
        ax.set_ylabel("Reward-design gain (acc(TRUE) - acc(SPURIOUS))")
        ax.set_title("Power: RD gain CI vs #seeds")
        ax.legend(fontsize=8)

        # Right: CI width vs n_seeds
        ax2 = axes[1]
        widths = [r["ci_width"] for r in rows]
        ax2.plot(ns, widths, "s-", color=plotstyle.COLORS["orange"])
        ax2.set_xlabel("Number of seeds")
        ax2.set_ylabel("95% CI width")
        ax2.set_title("CI width vs #seeds\n(power decreases with fewer seeds)")

        # Mark where CI excludes zero
        for i, r in enumerate(rows):
            marker = "v" if r["excludes_zero"] else "x"
            color  = plotstyle.COLORS["green"] if r["excludes_zero"] else plotstyle.COLORS["red"]
            ax.scatter(r["n_seeds"], r["ci_lo"] - 0.003, marker=marker,
                       color=color, s=60, zorder=5)

        fig.suptitle(f"Power Analysis (prior_strength={args.prior_strength:.2f})", fontsize=11)
        fig.tight_layout()
        return fig

    for base_dir in [FIGURES_DIR, os.path.join(RESULTS_DIR, "figures")]:
        plotstyle.save_fig(make_fig(), os.path.join(base_dir, "fig_power"))


if __name__ == "__main__":
    main()
