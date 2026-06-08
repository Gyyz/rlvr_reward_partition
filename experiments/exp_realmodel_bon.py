"""Experiment: REAL small-model best-of-N validation of the partition (GSM8K CoT).

The tabular-GRPO simulator is the primary instrument; here we validate that the
elicitation-vs-reward-design conflation appears in a REAL language model
(meta-llama/Llama-3.2-1B-Instruct) on a REAL reasoning benchmark (GSM8K), using
best-of-N selection as a one-step / reranking proxy for RLVR. We label this a
*different* mechanism (selection, not policy-gradient RL) but it exposes the same
telescoping decomposition.

Per-problem we draw N chain-of-thought samples and map:
  pass@1  (mean per-sample accuracy)        <-> RANDOM/FROZEN baseline a_R
  maj@N   (self-consistency majority vote)   <-> SPURIOUS elicitation   a_S
  bestN   (oracle: correct if any sample)    <-> TRUE verifier reward    a_T
with elicitation = a_S - a_R, reward_design = a_T - a_S, naive = a_T - a_R,
and RD fraction = reward_design / naive.

PRIOR STRENGTH (the sim's key axis) is recovered empirically and for free from a
SINGLE generation pass: we split problems by per-problem pass@1 into a
strong-prior subset (pass@1 >= median) and a weak-prior subset (pass@1 < median)
and report the decomposition within each. The prediction (from the simulator):
RD fraction is LOW on strong-prior problems (self-consistency already suffices)
and HIGH on weak-prior problems (the verifier is essential).

HEAVY experiment (torch + cached HF model + GSM8K). Invoked from
run_mess_compute_all.sh. Outputs: results/realmodel_bon.json,
../paper/figures/fig_realmodel_bon.{pdf,png}
"""
import sys, os, json, re, argparse
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import plotstyle
plotstyle.apply()
import matplotlib.pyplot as plt

CODE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(CODE_DIR, "results")
FIG_DIR = os.path.join(CODE_DIR, "..", "paper", "figures")


def gold_answer(ans):
    return int(ans.split("####")[-1].strip().replace(",", ""))


def extract_pred(text):
    # prefer "answer is <num>"; fall back to last integer
    m = re.search(r"answer is\D*(-?[\d,]+)", text, re.IGNORECASE)
    if not m:
        nums = re.findall(r"-?\d[\d,]*", text)
        if not nums:
            return None
        m_val = nums[-1]
    else:
        m_val = m.group(1)
    try:
        return int(m_val.replace(",", ""))
    except ValueError:
        return None


def decompose(pass_rates, maj_correct, best_correct):
    a_R = float(np.mean(pass_rates))
    a_S = float(np.mean(maj_correct))
    a_T = float(np.mean(best_correct))
    elicit, rd, naive = a_S - a_R, a_T - a_S, a_T - a_R
    rd_frac = rd / naive if abs(naive) > 1e-9 else float("nan")
    return {"pass@1": a_R, "maj@N": a_S, "bestN": a_T, "elicitation": elicit,
            "reward_design": rd, "naive": naive, "rd_fraction": rd_frac,
            "n": len(pass_rates)}


def main(model_id, n_problems, N, seed, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    torch.set_num_threads(16); torch.manual_seed(seed)
    os.makedirs(RESULTS_DIR, exist_ok=True); os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(os.path.join(RESULTS_DIR, "figures"), exist_ok=True)

    print(f"Loading {model_id} and GSM8K ...")
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32); model.eval()
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = np.random.default_rng(seed).choice(len(ds), size=n_problems, replace=False)

    per = []  # per-problem dict
    for j, ix in enumerate(idx):
        q = ds[int(ix)]["question"]; ans = gold_answer(ds[int(ix)]["answer"])
        msgs = [{"role": "user", "content": q + "\nThink step by step and end with 'The answer is <number>.'"}]
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids = tok(p, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=True,
                                 temperature=0.8, top_p=0.95, num_return_sequences=N,
                                 pad_token_id=tok.eos_token_id)
        preds = [extract_pred(tok.decode(out[i, ids["input_ids"].shape[1]:], skip_special_tokens=True))
                 for i in range(N)]
        correct = [pp == ans for pp in preds]
        valid = [pp for pp in preds if pp is not None]
        maj_ok = 0
        if valid:
            v, c = np.unique(valid, return_counts=True)
            maj_ok = int(int(v[np.argmax(c)]) == ans)
        per.append({"pass": float(np.mean(correct)), "maj": maj_ok, "best": int(any(correct))})
        if (j + 1) % 5 == 0:
            print(f"  {j+1}/{n_problems} done (running pass@1={np.mean([x['pass'] for x in per]):.3f})")

    pr = np.array([x["pass"] for x in per]); mj = np.array([x["maj"] for x in per]); bs = np.array([x["best"] for x in per])
    overall = decompose(pr, mj, bs)
    # split by empirical prior strength (per-problem pass@1)
    med = np.median(pr)
    strong = pr >= med if (pr >= med).sum() >= 3 else pr > pr.min()
    weak = ~strong
    strong_d = decompose(pr[strong], mj[strong], bs[strong]) if strong.sum() else None
    weak_d = decompose(pr[weak], mj[weak], bs[weak]) if weak.sum() else None

    out = {"model": model_id, "dataset": "gsm8k", "n_problems": n_problems, "N": N,
           "seed": seed, "max_new_tokens": max_new_tokens,
           "overall": overall, "strong_prior": strong_d, "weak_prior": weak_d,
           "median_pass1_split": float(med)}
    json.dump(out, open(os.path.join(RESULTS_DIR, "realmodel_bon.json"), "w"), indent=2)
    print("OVERALL:", {k: round(v, 3) for k, v in overall.items()})
    if strong_d: print("STRONG-prior subset:", {k: round(v, 3) for k, v in strong_d.items()})
    if weak_d:   print("WEAK-prior subset:  ", {k: round(v, 3) for k, v in weak_d.items()})

    # ---- figure ----
    def make_fig():
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.1))
        ax = axes[0]
        labels = ["pass@1\n(a_R)", "maj@N\n(a_S)", "best-of-N\n(a_T)"]
        vals = [overall["pass@1"], overall["maj@N"], overall["bestN"]]
        cols = [plotstyle.COLORS["orange"], plotstyle.COLORS["green"], plotstyle.COLORS["blue"]]
        ax.bar(range(3), vals, color=cols, edgecolor="k", linewidth=0.5)
        for i, v in enumerate(vals):
            ax.text(i, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(f"GSM8K accuracy (N={N})")
        ax.set_title(f"(a) {model_id.split('/')[-1]} on GSM8K\nelicit={overall['elicitation']:+.2f}, "
                     f"reward-design={overall['reward_design']:+.2f}")
        ax.set_ylim(0, 1.0)
        ax.annotate("", xy=(1, overall["maj@N"]), xytext=(0, overall["pass@1"]),
                    arrowprops=dict(arrowstyle="->", color=plotstyle.COLORS["green"]))
        ax.annotate("", xy=(2, overall["bestN"]), xytext=(1, overall["maj@N"]),
                    arrowprops=dict(arrowstyle="->", color=plotstyle.COLORS["blue"]))

        # Panel (b): decompose the naive "verifier gain" into elicitation + reward-design
        ax2 = axes[1]
        el = overall["elicitation"]; rd = overall["reward_design"]; naive = overall["naive"]
        if abs(naive) < 1e-9:
            naive = 1e-9
        ax2.bar([0], [el], color=plotstyle.COLORS["green"], edgecolor="k",
                label=f"Elicitation (self-consistency)  {el/naive*100:.0f}%")
        ax2.bar([0], [rd], bottom=[el], color=plotstyle.COLORS["blue"], edgecolor="k",
                label=f"Reward design (verifier)  {rd/naive*100:.0f}%")
        ax2.text(0, el / 2, f"+{el:.2f}", ha="center", va="center", fontsize=10, color="white")
        ax2.text(0, el + rd / 2, f"+{rd:.2f}", ha="center", va="center", fontsize=10, color="white")
        ax2.set_xlim(-0.8, 0.8); ax2.set_xticks([0])
        ax2.set_xticklabels([f"naive 'verifier gain'\n= best-of-N $-$ pass@1 = {naive:.2f}"], fontsize=9)
        ax2.set_ylabel("Accuracy gain over pass@1")
        ax2.set_title("(b) The naive gain conflates two mechanisms")
        ax2.legend(fontsize=8.5, loc="upper right")
        ax2.set_ylim(0, max(naive * 1.25, 0.05))
        fig.suptitle("Real-Model Best-of-N Validation of the Telescoping Partition (GSM8K)", fontsize=12, y=1.02)
        fig.tight_layout()
        return fig

    plotstyle.save_fig(make_fig(), os.path.join(FIG_DIR, "fig_realmodel_bon"))
    plotstyle.save_fig(make_fig(), os.path.join(RESULTS_DIR, "figures", "fig_realmodel_bon"))
    print("  figures written")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--n_problems", type=int, default=24)
    p.add_argument("--N", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_new_tokens", type=int, default=170)
    a = p.parse_args()
    main(a.model, a.n_problems, a.N, a.seed, a.max_new_tokens)
