"""Experiment: Re-audit of named RLVR results.

Simulates two "named results" from the literature and applies the re-audit
instrument to attribute the gain to FE vs RD:

1. "Qwen-math-like (strong prior)" — large total gain, mostly filtering.
2. "OLMo/Llama-like (weak prior)"  — smaller total gain, mostly reward-design.

Outputs
-------
results/reaudit.json
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.audit import audit_canonical_families

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",        type=int, default=0)
    parser.add_argument("--n_seeds",     type=int, default=10)
    parser.add_argument("--n_steps",     type=int, default=200)
    parser.add_argument("--n_prompts",   type=int, default=40)
    parser.add_argument("--n_actions",   type=int, default=8)
    parser.add_argument("--G",           type=int, default=8)
    parser.add_argument("--lr",          type=float, default=0.05)
    parser.add_argument("--beta",        type=float, default=0.01)
    parser.add_argument("--batch_size",  type=int, default=16)
    parser.add_argument("--n_bootstrap", type=int, default=300)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Running re-audit on canonical families ...")
    results = audit_canonical_families(
        n_seeds=args.n_seeds,
        n_steps=args.n_steps,
        base_seed=args.seed,
        n_prompts=args.n_prompts,
        n_actions=args.n_actions,
        G=args.G,
        lr=args.lr,
        beta=args.beta,
        batch_size=args.batch_size,
        n_bootstrap=args.n_bootstrap,
    )

    out = [r.to_dict() for r in results]
    out_path = os.path.join(RESULTS_DIR, "reaudit.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  saved {out_path}")


if __name__ == "__main__":
    main()
