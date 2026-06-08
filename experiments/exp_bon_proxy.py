"""Optional: Best-of-N proxy experiment on gpt2/distilgpt2.

*** THIS IS NOT REAL RLVR. ***
It illustrates the 'spurious rewards surface latent prior' phenomenon in a
Best-of-N (BoN) setting: we sample N completions from a *frozen* language
model and keep the one with the highest reward.  With random rewards, BoN
still gains over greedy decoding because BoN exploits the model's latent
prior distribution.

This is labelled as BoN-as-RL-proxy throughout, and is a *separate*
phenomenon from GRPO-style RL fine-tuning.

Skipped gracefully if transformers is unavailable or the model cannot be
loaded without network access.
"""

from __future__ import annotations

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Guard: try importing transformers
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch  # type: ignore
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "paper", "figures")

TOY_ARITHMETIC = [
    {"prompt": "What is 3 + 5?",   "answer": "8"},
    {"prompt": "What is 7 - 2?",   "answer": "5"},
    {"prompt": "What is 4 * 3?",   "answer": "12"},
    {"prompt": "What is 10 / 2?",  "answer": "5"},
    {"prompt": "What is 6 + 9?",   "answer": "15"},
]


def main():
    if not HAS_TRANSFORMERS:
        print("  [SKIP] transformers not available.")
        sys.exit(1)

    import numpy as np

    print("  Loading distilgpt2 (CPU) ...")
    try:
        tokenizer = AutoTokenizer.from_pretrained("distilgpt2", local_files_only=False)
        model = AutoModelForCausalLM.from_pretrained(
            "distilgpt2", local_files_only=False
        )
        model.eval()
    except Exception as e:
        print(f"  [SKIP] Cannot load distilgpt2: {e}")
        sys.exit(1)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    rng = np.random.default_rng(42)
    N_list = [1, 2, 4, 8, 16]
    rows = []

    for task in TOY_ARITHMETIC:
        prompt = task["prompt"]
        answer = task["answer"]

        for N in N_list:
            inputs = tokenizer(prompt + " Answer:", return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=5,
                    do_sample=True,
                    temperature=1.0,
                    num_return_sequences=N,
                    pad_token_id=tokenizer.eos_token_id,
                )
            texts = [
                tokenizer.decode(o[inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
                for o in outputs
            ]

            # True reward: does any completion contain the correct answer?
            true_any = any(answer in t for t in texts)
            # Random reward: pick a random one, check if it happens to be correct
            rand_pick = texts[int(rng.integers(0, len(texts)))]
            rand_correct = answer in rand_pick

            rows.append({
                "prompt": prompt,
                "answer": answer,
                "N": N,
                "bon_true": int(true_any),
                "bon_random": int(rand_correct),
            })

    # Aggregate by N
    agg = {}
    for r in rows:
        n = r["N"]
        if n not in agg:
            agg[n] = {"true": [], "random": []}
        agg[n]["true"].append(r["bon_true"])
        agg[n]["random"].append(r["bon_random"])

    summary = []
    for n in N_list:
        t = np.mean(agg[n]["true"])
        r = np.mean(agg[n]["random"])
        print(f"  BoN N={n:2d}  acc(true)={t:.3f}  acc(random)={r:.3f}  "
              f"gain_true={t - agg[1]['true'][0]:.3f}")
        summary.append({"N": n, "acc_true": float(t), "acc_random": float(r)})

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "bon_proxy.json")
    with open(out_path, "w") as f:
        json.dump({
            "note": "BoN-as-RL-proxy — NOT real RLVR. Uses frozen distilgpt2.",
            "summary": summary,
        }, f, indent=2)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
