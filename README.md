# RLVR Reward Partition: "What Does 'True Minus Random' Estimate?"

A controlled, CPU-only synthetic simulator for studying the causal structure
of RLVR (Reinforcement Learning from Verifiable Rewards) gains.

## Scientific question

Recent work shows that training with *random* (Bernoulli-noise) rewards still
improves strong-prior models (the "Spurious Rewards" phenomenon, arXiv:2506.10947).
We ask: **what does the common diagnostic `acc(TRUE) - acc(RANDOM)` actually
estimate?**  We propose a causal partition into two terms:

* **FE (Filtering+Elicitation) gain** = `acc(RANDOM) - acc(FROZEN)` — the gain
  from the structural/algorithmic aspects of RLVR (dynamic-sampling filtering,
  KL regularisation), which surface the latent prior even with non-informative
  rewards.
* **RD (Reward Design) gain** = `acc(TRUE) - acc(RANDOM)` — the residual gain
  from having a genuinely informative reward signal.

We validate the partition empirically via a 2×2×2 factorial experiment and
an additivity specification test, with a pre-registered invalidation threshold.

## Repository layout

```
code/
  src/
    env.py        — Synthetic task distribution + reward functions
    grpo.py       — Tabular GRPO implementation
    partition.py  — THE CONTRIBUTION: FE/RD partition, 2x2x2 factorial
    audit.py      — Re-audit instrument for reported RLVR results
  experiments/
    exp_prior_sweep.py
    exp_factorial.py
    exp_points_vs_bounds.py
    exp_reaudit.py
    exp_power.py
    exp_bon_proxy.py          (optional, requires transformers)
    run_all_experiments.sh    (LIGHT, < 4 min)
    run_mess_compute_all.sh   (HEAVY, 30-120 min)
  tests/
    test_env.py
    test_grpo.py
  results/
    *.json                    (produced by run_all_experiments.sh)
    figures/                  (local copy of paper figures)
  plotstyle.py
  requirements.txt
  README.md
```

## Quickstart

```bash
# Run all light experiments (< 4 min on CPU)
bash experiments/run_all_experiments.sh

# Run sanity tests
python tests/test_env.py
python tests/test_grpo.py
```

## Key findings

1. **Prior-sweep**: `FE gain` rises monotonically with `prior_strength` while
   `RD gain` shrinks. This reproduces and explains the Spurious Rewards
   phenomenon: strong-prior models gain more from random rewards simply because
   filtering surfaces their already-good prior.

2. **2×2×2 Factorial**: The linear model identifies `C (prior_strength)` and
   `B (filtering)` as strong main effects. The interaction `BC` is the largest
   interaction term, representing the "spurious rewards" coupling.

3. **Additivity spec test**: For typical settings, the interaction ratio is
   below the pre-registered threshold (0.20), supporting POINT estimates.
   Format-boosted priors can push above the threshold, yielding BOUNDS.

4. **Re-audit**: Qwen-math-like (strong-prior) families are
   `FILTERING_DOMINATED`; OLMo/Llama-like (weak-prior) families are
   `REWARD_DESIGN_DOMINATED`.

## Environment

Python 3.11, CPU-only, Linux.
Dependencies: numpy 1.26, scipy 1.13, scikit-learn 1.5, matplotlib 3.8.
No torch, no GPU required for the core experiments.
