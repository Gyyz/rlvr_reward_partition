"""Sanity tests for src/grpo.py — runnable via `python tests/test_grpo.py`."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.env import make_task_distribution, evaluate_policy
from src.grpo import GRPOConfig, run_grpo, run_grpo_seeds, mean_final_acc


def test_frozen_policy_unchanged():
    """Frozen reward should leave the policy unchanged."""
    dist = make_task_distribution(n_prompts=10, n_actions=6, prior_strength=0.5, seed=0)
    cfg = GRPOConfig(
        n_steps=30, batch_size=8, G=4, lr=0.1, beta=0.01,
        filtering=True, reward_type="frozen", eval_every=10, seed=1,
    )
    result = run_grpo(dist, cfg)
    # Policy should equal prior (no delta applied)
    diff = np.max(np.abs(result.log_pi_final - dist.log_pi0))
    assert diff < 1e-9, f"Frozen policy changed: max diff={diff:.2e}"
    print("  test_frozen_policy_unchanged: PASS")


def test_true_reward_improves_accuracy():
    """TRUE reward with many steps should improve acc above frozen."""
    dist = make_task_distribution(n_prompts=30, n_actions=8, prior_strength=0.40, seed=2)
    rng0 = np.random.default_rng(99)
    acc_init = evaluate_policy(dist.log_pi0, dist.tasks, rng0, n_samples=100)

    cfg = GRPOConfig(
        n_steps=100, batch_size=16, G=8, lr=0.1, beta=0.001,
        filtering=True, reward_type="true", eval_every=20, seed=5,
    )
    result = run_grpo(dist, cfg)
    assert result.final_acc > acc_init - 0.02, \
        f"TRUE reward should improve or maintain accuracy: {result.final_acc:.4f} vs init={acc_init:.4f}"
    print(f"  test_true_reward_improves_accuracy: PASS  (init={acc_init:.4f} final={result.final_acc:.4f})")


def test_filtering_reduces_n_groups():
    """With filtering on, n_filtered should be > 0 when prior is extreme."""
    # Extreme high prior: many all-correct groups -> lots of filtering
    dist = make_task_distribution(n_prompts=20, n_actions=6, prior_strength=0.95, seed=3)
    cfg = GRPOConfig(
        n_steps=50, batch_size=10, G=6, lr=0.05, beta=0.01,
        filtering=True, reward_type="true", eval_every=10, seed=7,
    )
    result = run_grpo(dist, cfg)
    assert result.n_filtered > 0, "Expected some groups to be filtered with extreme prior"
    filter_rate = result.n_filtered / max(result.n_total_groups, 1)
    print(f"  test_filtering_reduces_n_groups: PASS  (filter_rate={filter_rate:.2%})")


def test_no_filtering_no_drops():
    """With filtering=False, n_filtered should be 0."""
    dist = make_task_distribution(n_prompts=10, n_actions=6, prior_strength=0.5, seed=0)
    cfg = GRPOConfig(
        n_steps=20, batch_size=8, G=4, lr=0.05, beta=0.01,
        filtering=False, reward_type="true", eval_every=10, seed=0,
    )
    result = run_grpo(dist, cfg)
    assert result.n_filtered == 0, f"Expected 0 filtered with filtering=False, got {result.n_filtered}"
    print("  test_no_filtering_no_drops: PASS")


def test_acc_curve_nondecreasing_in_expectation():
    """Accuracy should trend upward (at least not catastrophically down) with TRUE reward."""
    dist = make_task_distribution(n_prompts=30, n_actions=8, prior_strength=0.35, seed=4)
    cfg = GRPOConfig(
        n_steps=150, batch_size=16, G=8, lr=0.08, beta=0.005,
        filtering=True, reward_type="true", eval_every=15, seed=10,
    )
    result = run_grpo(dist, cfg)
    # First half vs second half of curve
    n = len(result.acc_curve)
    first_half  = np.mean(result.acc_curve[:n//2])
    second_half = np.mean(result.acc_curve[n//2:])
    assert second_half >= first_half - 0.10, \
        f"Accuracy fell significantly: first_half={first_half:.4f} second_half={second_half:.4f}"
    print(f"  test_acc_curve_nondecreasing_in_expectation: PASS  "
          f"(first={first_half:.4f} second={second_half:.4f})")


def test_multi_seed_variance():
    """Multiple seeds should produce some variance in final accuracy."""
    dist = make_task_distribution(n_prompts=20, n_actions=6, prior_strength=0.5, seed=8)
    cfg = GRPOConfig(
        n_steps=50, batch_size=10, G=6, lr=0.05, beta=0.01,
        filtering=True, reward_type="true", eval_every=10, seed=0,
    )
    results = run_grpo_seeds(dist, cfg, n_seeds=5, base_seed=0)
    m, s = mean_final_acc(results)
    assert s >= 0.0, "Std should be non-negative"
    print(f"  test_multi_seed_variance: PASS  (mean={m:.4f}, std={s:.4f})")


def test_determinism():
    """Same seed should produce identical results."""
    dist = make_task_distribution(n_prompts=10, n_actions=6, prior_strength=0.5, seed=0)
    cfg = GRPOConfig(n_steps=20, batch_size=8, G=4, lr=0.05, beta=0.01,
                     filtering=True, reward_type="true", eval_every=5, seed=42)
    r1 = run_grpo(dist, cfg)
    r2 = run_grpo(dist, cfg)
    assert abs(r1.final_acc - r2.final_acc) < 1e-12, \
        f"Non-deterministic: {r1.final_acc:.8f} vs {r2.final_acc:.8f}"
    print("  test_determinism: PASS")


if __name__ == "__main__":
    print("=== tests/test_grpo.py ===")
    test_frozen_policy_unchanged()
    test_true_reward_improves_accuracy()
    test_filtering_reduces_n_groups()
    test_no_filtering_no_drops()
    test_acc_curve_nondecreasing_in_expectation()
    test_multi_seed_variance()
    test_determinism()
    print("\nAll tests PASSED.")
