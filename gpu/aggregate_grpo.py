"""Aggregate the real GRPO runs into the telescoping decomposition + figure."""
import os, json, glob
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle
plotstyle.apply()
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
OUT_FIG = os.path.join(CODE, "..", "paper", "figures")
RES = os.path.join(CODE, "results")

MODELS = [("Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B\n(strong prior)"),
          ("Llama-3.2-1B-Instruct", "Llama-3.2-1B\n(weak prior)")]


def load(model, reward):
    p = os.path.join(HERE, "results_grpo", f"grpo_{model}_{reward}.json")
    return json.load(open(p))


def main():
    rows = {}
    for m, _ in MODELS:
        aF = load(m, "frozen")["final_acc"]
        aR = load(m, "random")["final_acc"]
        aS = load(m, "spurious")["final_acc"]
        aT = load(m, "true")["final_acc"]
        elicit = aS - aR
        rd = aT - aS
        naive = aT - aR
        null = aR - aF
        rows[m] = {"a_frozen": aF, "a_random": aR, "a_spurious": aS, "a_true": aT,
                   "random_null": null, "elicitation": elicit, "reward_design": rd,
                   "naive": naive, "rd_fraction": (rd / naive if abs(naive) > 1e-9 else float("nan"))}
    os.makedirs(RES, exist_ok=True)
    json.dump(rows, open(os.path.join(RES, "grpo_real.json"), "w"), indent=2)
    for m, _ in MODELS:
        r = rows[m]
        print(f"{m}: F={r['a_frozen']:.3f} R={r['a_random']:.3f} S={r['a_spurious']:.3f} T={r['a_true']:.3f} "
              f"| null={r['random_null']:+.3f} elicit={r['elicitation']:+.3f} rd={r['reward_design']:+.3f} "
              f"naive={r['naive']:+.3f} RDfrac={r['rd_fraction']:.2f}")

    def make_fig():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        # (a) grouped bars: 4 conditions x 2 models
        ax = axes[0]
        conds = [("a_frozen", "frozen", "gray"), ("a_random", "random", "orange"),
                 ("a_spurious", "self-cons.", "green"), ("a_true", "true", "blue")]
        x = np.arange(len(MODELS)); w = 0.2
        for j, (key, lab, col) in enumerate(conds):
            ax.bar(x + (j - 1.5) * w, [rows[m][key] for m, _ in MODELS], w,
                   color=plotstyle.COLORS[col], edgecolor="k", linewidth=0.4, label=lab)
        ax.set_xticks(x); ax.set_xticklabels([lbl for _, lbl in MODELS], fontsize=9)
        ax.set_ylabel("GSM8K pass@1 after GRPO")
        ax.set_title("(a) Real GRPO: four reward conditions")
        ax.legend(fontsize=8, ncol=2); ax.set_ylim(0, 0.7)
        # (b) decomposition: elicitation (sign-flips) vs reward-design
        ax2 = axes[1]
        el = [rows[m]["elicitation"] for m, _ in MODELS]
        rd = [rows[m]["reward_design"] for m, _ in MODELS]
        ax2.bar(x - w/1.6, el, w*1.2, color=plotstyle.COLORS["green"], edgecolor="k", label="Elicitation (S$-$R)")
        ax2.bar(x + w/1.6, rd, w*1.2, color=plotstyle.COLORS["blue"], edgecolor="k", label="Reward design (T$-$S)")
        ax2.axhline(0, color="k", lw=0.8)
        ax2.set_xticks(x); ax2.set_xticklabels([lbl for _, lbl in MODELS], fontsize=9)
        ax2.set_ylabel("Accuracy gain")
        ax2.set_title("(b) Elicitation sign-flips with prior strength")
        for i, (e, r) in enumerate(zip(el, rd)):
            ax2.text(i - w/1.6, e + (0.005 if e >= 0 else -0.012), f"{e:+.2f}", ha="center", fontsize=8)
            ax2.text(i + w/1.6, r + 0.005, f"{r:+.2f}", ha="center", fontsize=8)
        ax2.legend(fontsize=8)
        fig.suptitle("Real GRPO RLVR on GSM8K: the Partition Holds Under Genuine RL", fontsize=12, y=1.02)
        fig.tight_layout(); return fig

    plotstyle.save_fig(make_fig(), os.path.join(OUT_FIG, "fig_grpo_real"))
    plotstyle.save_fig(make_fig(), os.path.join(RES, "figures", "fig_grpo_real"))
    print("  figure written")


if __name__ == "__main__":
    os.makedirs(os.path.join(RES, "figures"), exist_ok=True)
    main()
