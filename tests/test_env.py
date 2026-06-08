"""Sanity tests for src/env.py — runnable via `python tests/test_env.py`."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.env import (
    make_task_distribution,
    evaluate_policy,
    passk,
    reward_true,
    reward_random,
    reward_format,
    reward_frozen,
    _logsumexp,
)


def test_logsumexp():
    logits = np.array([1.0, 2.0, 3.0])
    expected = np.log(np.sum(np.exp(logits)))
    assert abs(_logsumexp(logits) - expected) < 1e-9, "logsumexp incorrect"
    print("  test_logsumexp: PASS")


def test_task_distribution_shape():
    dist = make_task_distribution(n_prompts=10, n_actions=6, prior_strength=0.7, seed=0)
    assert dist.log_pi0.shape == (10, 6), "log_pi0 shape mismatch"
    assert dist.n_prompts == 10
    assert dist.n_actions == 6
    print("  test_task_distribution_shape: PASS")


def test_prior_log_probs_sum_to_one():
    dist = make_task_distribution(n_prompts=5, n_actions=6, prior_strength=0.6, seed=1)
    for p in range(dist.n_prompts):
        lp = dist.log_pi0[p]
        total = np.sum(np.exp(lp))
        assert abs(total - 1.0) < 1e-6, f"probs don't sum to 1 for prompt {p}: {total}"
    print("  test_prior_log_probs_sum_to_one: PASS")


def test_prior_strength_ordering():
    """Higher prior_strength should give higher pass@1 from the prior."""
    dist_lo = make_task_distribution(n_prompts=50, n_actions=8, prior_strength=0.30, seed=3)
    dist_hi = make_task_distribution(n_prompts=50, n_actions=8, prior_strength=0.80, seed=3)
    rng = np.random.default_rng(99)
    acc_lo = evaluate_policy(dist_lo.log_pi0, dist_lo.tasks, rng, n_samples=200)
    rng = np.random.default_rng(99)
    acc_hi = evaluate_policy(dist_hi.log_pi0, dist_hi.tasks, rng, n_samples=200)
    assert acc_hi > acc_lo, f"Higher prior should give higher acc: {acc_hi:.4f} vs {acc_lo:.4f}"
    print(f"  test_prior_strength_ordering: PASS  (acc_hi={acc_hi:.4f} > acc_lo={acc_lo:.4f})")


def test_frozen_reward_always_zero():
    task = make_task_distribution(n_prompts=3, n_actions=4, seed=0).tasks[0]
    for a in range(task.n_actions):
        assert reward_frozen(task, a) == 0.0, "frozen reward should always be 0"
    print("  test_frozen_reward_always_zero: PASS")


def test_true_reward_correctness():
    dist = make_task_distribution(n_prompts=3, n_actions=6, prior_strength=0.5, seed=5)
    task = dist.tasks[0]
    for a in range(task.n_actions):
        r = reward_true(task, a)
        assert r == float(task.correct_mask[a]), f"reward_true mismatch at action {a}"
    print("  test_true_reward_correctness: PASS")


def test_format_action_not_correct():
    """Format action should NOT be correct (by construction)."""
    dist = make_task_distribution(n_prompts=20, n_actions=8, seed=7)
    for task in dist.tasks:
        assert not task.correct_mask[task.format_action], \
            f"format_action={task.format_action} is incorrectly marked correct"
    print("  test_format_action_not_correct: PASS")


def test_random_reward_is_bernoulli():
    """Random reward should be ~0.5 over many calls."""
    task = make_task_distribution(n_prompts=1, n_actions=4, seed=0).tasks[0]
    rng = np.random.default_rng(0)
    rewards = [reward_random(task, 0, rng) for _ in range(1000)]
    mean_r = np.mean(rewards)
    assert abs(mean_r - 0.5) < 0.05, f"random reward mean={mean_r:.3f} not ~0.5"
    print(f"  test_random_reward_is_bernoulli: PASS  (mean={mean_r:.3f})")


def test_passk_monotone_in_k():
    """pass@k should be non-decreasing in k."""
    dist = make_task_distribution(n_prompts=20, n_actions=6, prior_strength=0.5, seed=2)
    p1 = passk(dist.log_pi0, dist.tasks, k=1)
    p4 = passk(dist.log_pi0, dist.tasks, k=4)
    p8 = passk(dist.log_pi0, dist.tasks, k=8)
    assert p1 <= p4 <= p8, f"pass@k not monotone: {p1:.4f} {p4:.4f} {p8:.4f}"
    print(f"  test_passk_monotone_in_k: PASS  (p@1={p1:.4f} p@4={p4:.4f} p@8={p8:.4f})")


if __name__ == "__main__":
    print("=== tests/test_env.py ===")
    test_logsumexp()
    test_task_distribution_shape()
    test_prior_log_probs_sum_to_one()
    test_prior_strength_ordering()
    test_frozen_reward_always_zero()
    test_true_reward_correctness()
    test_format_action_not_correct()
    test_random_reward_is_bernoulli()
    test_passk_monotone_in_k()
    print("\nAll tests PASSED.")
