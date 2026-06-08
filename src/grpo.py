"""Tabular GRPO implementation for the synthetic RLVR simulator.

This module implements Group Relative Policy Optimisation (GRPO) on a
tabular softmax policy.  The policy is represented as a (n_prompts x
n_actions) log-probability matrix; each "generation" is a single discrete
action sampled from the policy distribution.

Key design choices
------------------
* Group normalisation: advantages within each group of G completions are
  normalised to zero mean and unit variance (the GRPO "group-normalised"
  advantage), matching the description in Shao et al. (2024).
* Filtering (dynamic sampling): if ``filtering=True``, groups where ALL G
  completions received the same reward (all-correct OR all-wrong) are
  dropped before the policy-gradient update.  This is the key mechanism that
  "surfaces the latent prior" without informative reward.
* KL-to-prior penalty: a per-prompt KL(pi || pi0) term is added with weight
  beta, matching the RLVR training objective.
* Frozen mode: when the reward function is ``reward_frozen``, the policy
  is never updated (accumulates the evaluation at each step but the log_pi
  matrix is not touched).  This is the cleanest way to estimate the
  "no-training" baseline.

The update rule (simplified)
-----------------------------
For each prompt p in a mini-batch:
  1. Sample G actions a_{g} ~ pi_theta(· | p).
  2. Compute rewards r_{g} = R(task_p, a_{g}).
  3. If filtering: skip this prompt if all r_{g} are identical.
  4. Compute group-normalised advantage:
         A_{g} = (r_{g} - mean(r)) / (std(r) + eps)
  5. Policy-gradient gradient on the tabular policy (exact):
         delta_logit[p, a_{g}] += lr * A_{g}   (accumulate)
  6. KL penalty: gradient of -beta * KL(pi || pi0) w.r.t. logits:
         g_kl[p, :] = -beta * (pi - pi0)
  7. logits[p] += lr * (pg_grad[p] + g_kl[p])

Tabular policy gradient
-----------------------
For a softmax policy pi(a) = softmax(logit[a]):
  d/d logit[a] log pi(a_g) = I[a == a_g] - pi(a)
So the policy-gradient gradient from one sample (a_g, A_g) is:
  g_a = A_g * (e_{a_g} - pi)
where e_{a_g} is the one-hot vector.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional

from .env import (
    Task,
    TaskDistribution,
    REWARD_FUNCTIONS,
    _logsumexp,
    evaluate_policy,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class GRPOConfig:
    """Hyper-parameters for GRPO training.

    Parameters
    ----------
    n_steps : int
        Number of GRPO update steps (each step = one mini-batch of prompts).
    batch_size : int
        Prompts sampled per step.
    G : int
        Completions sampled per prompt per step (group size).
    lr : float
        Policy learning rate (logit update step size).
    beta : float
        KL-to-prior regularisation coefficient.
    filtering : bool
        Whether to apply dynamic-sampling filtering (drop all-correct /
        all-wrong groups before the PG update).
    reward_type : str
        One of "true", "random", "format", "frozen".
    eval_every : int
        Evaluate pass@1 every this many steps.
    eval_samples : int
        Monte Carlo samples per prompt during evaluation.
    seed : int
        RNG seed for reproducibility.
    eps : float
        Small constant for advantage normalisation stability.
    """
    n_steps: int = 300
    batch_size: int = 16
    G: int = 8
    lr: float = 0.05
    beta: float = 0.01
    filtering: bool = True
    reward_type: str = "true"
    eval_every: int = 20
    eval_samples: int = 64
    seed: int = 0
    eps: float = 1e-8


# ---------------------------------------------------------------------------
# Training results
# ---------------------------------------------------------------------------

@dataclass
class TrainingResult:
    """Results returned by ``run_grpo``.

    Attributes
    ----------
    final_acc : float
        Pass@1 accuracy at the end of training.
    acc_curve : list[float]
        Accuracy evaluated every ``eval_every`` steps.
    step_curve : list[int]
        Corresponding step indices.
    log_pi_final : np.ndarray
        Final log-probability matrix.
    n_filtered : int
        Total number of groups dropped by filtering.
    n_total_groups : int
        Total groups considered before filtering.
    config : GRPOConfig
        The configuration used for this run.
    """
    final_acc: float
    acc_curve: list[float]
    step_curve: list[int]
    log_pi_final: np.ndarray
    n_filtered: int
    n_total_groups: int
    config: GRPOConfig


# ---------------------------------------------------------------------------
# Core training loop
# ---------------------------------------------------------------------------

def run_grpo(
    dist: TaskDistribution,
    config: GRPOConfig,
) -> TrainingResult:
    """Run GRPO on a tabular policy and return results.

    Parameters
    ----------
    dist : TaskDistribution
        Task distribution (contains tasks + prior policy log_pi0).
    config : GRPOConfig
        Training configuration.

    Returns
    -------
    TrainingResult
    """
    rng = np.random.default_rng(config.seed)

    # Clone the prior as the starting policy (logit = log_pi0 + const)
    # We work in logit space (unnormalised); we normalise for sampling.
    logits = dist.log_pi0.copy()  # shape (n_prompts, n_actions)
    log_pi0 = dist.log_pi0.copy()  # reference prior (never modified)

    # "spurious" is a GROUP-level (self-consistency / majority-vote) reward and
    # is therefore computed inside the rollout loop, not via a per-action fn.
    reward_fn = REWARD_FUNCTIONS.get(config.reward_type, None)

    acc_curve: list[float] = []
    step_curve: list[int] = []
    n_filtered = 0
    n_total_groups = 0

    # Initial evaluation (step 0)
    acc0 = evaluate_policy(logits, dist.tasks, rng, config.eval_samples)
    acc_curve.append(acc0)
    step_curve.append(0)

    for step in range(1, config.n_steps + 1):
        # --- sample mini-batch of prompts -----------------------------------
        batch_indices = rng.choice(
            dist.n_prompts, size=min(config.batch_size, dist.n_prompts), replace=False
        )

        # Accumulate gradient updates (we apply them at the end of each step)
        delta = np.zeros_like(logits)

        for p in batch_indices:
            task = dist.tasks[p]

            # Compute current normalised probabilities for prompt p
            lp = logits[p]
            probs = _softmax(lp)

            # --- sample G completions ---------------------------------------
            actions = rng.choice(task.n_actions, size=config.G, p=probs)

            # --- compute rewards --------------------------------------------
            if config.reward_type == "spurious":
                # Self-consistency / majority-vote pseudo-reward (TTRL-style):
                # reward 1 for actions matching the plurality answer in THIS
                # group, 0 otherwise. Carries NO ground-truth correctness signal
                # but reinforces the policy's own modal answer -> elicits the
                # latent prior. Helps iff the prior's mode is correct.
                vals, counts = np.unique(actions, return_counts=True)
                plurality = int(vals[np.argmax(counts)])
                rewards = (actions == plurality).astype(float)
            elif config.reward_type == "random":
                rewards = np.array([reward_fn(task, a, rng) for a in actions], dtype=float)
            else:
                rewards = np.array([reward_fn(task, a) for a in actions], dtype=float)

            n_total_groups += 1

            # --- filtering --------------------------------------------------
            if config.filtering:
                r_std = rewards.std()
                if r_std < config.eps:
                    # all rewards identical -> drop group
                    n_filtered += 1
                    continue

            # --- group-normalised advantages --------------------------------
            r_mean = rewards.mean()
            r_std  = rewards.std() + config.eps
            advantages = (rewards - r_mean) / r_std

            # --- policy-gradient gradient -----------------------------------
            # d/d logit[a] log pi(a_g) = I[a == a_g] - pi(a)
            pg_grad = np.zeros(task.n_actions)
            for g_idx in range(config.G):
                a_g = actions[g_idx]
                A_g = advantages[g_idx]
                one_hot = np.zeros(task.n_actions)
                one_hot[a_g] = 1.0
                pg_grad += A_g * (one_hot - probs)
            pg_grad /= config.G  # average over group

            # --- KL gradient ------------------------------------------------
            # KL(pi || pi0) = sum_a pi(a) [log pi(a) - log pi0(a)]
            # d/d logit[a] KL = pi(a) - pi0(a)    (using softmax Jacobian)
            pi0 = _softmax(log_pi0[p])
            kl_grad = -(probs - pi0)  # negative because we want to MINIMISE KL

            # --- accumulate update ------------------------------------------
            if config.reward_type == "frozen":
                # Frozen: no update at all
                pass
            else:
                delta[p] += config.lr * (pg_grad + config.beta * kl_grad)

        # Apply accumulated delta
        logits += delta

        # --- periodic evaluation --------------------------------------------
        if step % config.eval_every == 0 or step == config.n_steps:
            acc = evaluate_policy(logits, dist.tasks, rng, config.eval_samples)
            acc_curve.append(acc)
            step_curve.append(step)

    final_acc = evaluate_policy(logits, dist.tasks, rng, config.eval_samples * 2)

    return TrainingResult(
        final_acc=final_acc,
        acc_curve=acc_curve,
        step_curve=step_curve,
        log_pi_final=logits.copy(),
        n_filtered=n_filtered,
        n_total_groups=n_total_groups,
        config=config,
    )


# ---------------------------------------------------------------------------
# Multi-seed runner
# ---------------------------------------------------------------------------

def run_grpo_seeds(
    dist: TaskDistribution,
    config: GRPOConfig,
    n_seeds: int = 8,
    base_seed: int = 0,
) -> list[TrainingResult]:
    """Run GRPO with n_seeds independent seeds and return all results.

    Parameters
    ----------
    dist : TaskDistribution
    config : GRPOConfig
        Base config; seed will be overridden for each run.
    n_seeds : int
    base_seed : int
        Seeds will be base_seed, base_seed+1, ..., base_seed+n_seeds-1.

    Returns
    -------
    list[TrainingResult]
    """
    results = []
    for i in range(n_seeds):
        cfg = GRPOConfig(
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            G=config.G,
            lr=config.lr,
            beta=config.beta,
            filtering=config.filtering,
            reward_type=config.reward_type,
            eval_every=config.eval_every,
            eval_samples=config.eval_samples,
            seed=base_seed + i,
            eps=config.eps,
        )
        results.append(run_grpo(dist, cfg))
    return results


def mean_final_acc(results: list[TrainingResult]) -> tuple[float, float]:
    """Return (mean, std) of final accuracy across seeds."""
    accs = np.array([r.final_acc for r in results])
    return float(accs.mean()), float(accs.std(ddof=1) if len(accs) > 1 else 0.0)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = logits - logits.max()
    e = np.exp(shifted)
    return e / e.sum()
