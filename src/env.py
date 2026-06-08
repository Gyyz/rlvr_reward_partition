"""Synthetic reasoning task environment for RLVR causal partition experiments.

Each "prompt" has a discrete action space of candidate answers.  Candidates
carry latent binary correctness.  A "base-model family" is described by a
tabular prior policy pi0 over (prompt, answer) pairs whose concentration on
correct answers is controlled by *prior_strength*.

Design decisions
----------------
* Fully tabular: policies are (n_prompts × n_actions) numpy arrays of
  log-probabilities, so a GRPO step is a single numpy operation — no neural
  network, no tokeniser, no GPU.
* Deterministic given a seed: all randomness flows through a single
  ``np.random.default_rng(seed)`` instance.
* Two canonical families:
  - STRONG_PRIOR  (default prior_strength=0.85): mimics Qwen-Math, which
    already solves many problems before RL.
  - WEAK_PRIOR    (default prior_strength=0.35): mimics Llama/OLMo.
* FORMAT prior: one dedicated "format-correct" action per prompt whose
  prior probability is controlled by ``format_strength``.  The true-reward
  function does NOT give extra credit for format (format is orthogonal to
  correctness) but a FORMAT reward function exclusively rewards this action,
  letting us test the FORMAT confound.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Task:
    """A single reasoning task / prompt with its candidate-answer structure.

    Attributes
    ----------
    prompt_id : int
        Unique identifier.
    n_actions : int
        Number of candidate answers (including the format-correct sentinel).
    correct_mask : np.ndarray, shape (n_actions,), dtype bool
        Which actions give a correct answer.
    format_action : int
        Index of the "format-correct" action (may or may not be correct).
    """
    prompt_id: int
    n_actions: int
    correct_mask: np.ndarray
    format_action: int


@dataclass
class TaskDistribution:
    """A collection of tasks with a shared prior policy.

    Attributes
    ----------
    tasks : list[Task]
    log_pi0 : np.ndarray, shape (n_prompts, max_actions)
        Log-probabilities of the prior policy.  Padded with -inf for
        prompt-specific action counts < max_actions if needed, but for
        simplicity we use uniform max_actions here.
    prior_strength : float
        Fraction of total probability mass the prior concentrates on
        correct actions (aggregated over all prompts).
    """
    tasks: list[Task]
    log_pi0: np.ndarray
    prior_strength: float

    @property
    def n_prompts(self) -> int:
        return len(self.tasks)

    @property
    def n_actions(self) -> int:
        return self.log_pi0.shape[1]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_task_distribution(
    n_prompts: int = 50,
    n_actions: int = 8,
    prior_strength: float = 0.65,
    format_strength: float = 0.20,
    n_correct_per_prompt: int = 2,
    seed: int = 0,
) -> TaskDistribution:
    """Build a synthetic task distribution.

    Parameters
    ----------
    n_prompts : int
        Number of independent prompts.
    n_actions : int
        Candidate actions per prompt (must be >= n_correct_per_prompt + 1).
    prior_strength : float in [0, 1]
        Aggregate probability mass on correct actions for the prior policy.
        0 = uniform; 1 = deterministic correct.
    format_strength : float in [0, 1]
        Additional probability mass given to the format-correct action on top
        of what it would receive from prior_strength (if it is also correct).
    n_correct_per_prompt : int
        Number of correct actions per prompt.
    seed : int
        RNG seed.

    Returns
    -------
    TaskDistribution
    """
    if n_actions < n_correct_per_prompt + 1:
        raise ValueError(
            f"n_actions={n_actions} must be > n_correct_per_prompt={n_correct_per_prompt}"
        )

    rng = np.random.default_rng(seed)

    tasks: list[Task] = []
    log_pi0 = np.zeros((n_prompts, n_actions))  # fill in below

    for p in range(n_prompts):
        # --- sample correctness mask ----------------------------------------
        correct_indices = rng.choice(n_actions, size=n_correct_per_prompt, replace=False)
        correct_mask = np.zeros(n_actions, dtype=bool)
        correct_mask[correct_indices] = True

        # format action: an action that is "well-formatted" but not necessarily
        # correct.  We pick one of the *incorrect* actions to be format-correct
        # so that format reward is clearly decoupled from correctness.
        incorrect_indices = np.where(~correct_mask)[0]
        format_action = int(rng.choice(incorrect_indices))

        tasks.append(Task(
            prompt_id=p,
            n_actions=n_actions,
            correct_mask=correct_mask,
            format_action=format_action,
        ))

        # --- build prior policy ---------------------------------------------
        # Start with uniform
        logits = np.zeros(n_actions)

        # Boost correct actions by a factor derived from prior_strength
        # prior_strength = P(correct) = sum_{correct} pi0(a)
        # For k correct actions out of n, uniform gives k/n.
        # We want to redistribute mass so correct actions get prior_strength
        # total and incorrect actions get 1 - prior_strength.
        k = n_correct_per_prompt
        n_wrong = n_actions - k
        if k > 0 and n_wrong > 0:
            # correct share per action vs wrong share per action
            correct_share = prior_strength / k
            wrong_share = (1.0 - prior_strength) / n_wrong
            correct_share = np.clip(correct_share, 1e-9, 1.0)
            wrong_share   = np.clip(wrong_share,   1e-9, 1.0)
            for a in range(n_actions):
                if correct_mask[a]:
                    logits[a] = np.log(correct_share)
                else:
                    logits[a] = np.log(wrong_share)

        # Boost format action: add extra log-mass proportional to format_strength
        if format_strength > 0:
            # We add a bonus to the format action, then renormalise
            logits[format_action] += np.log(1.0 + format_strength * n_actions)

        # Normalise to log-probabilities
        log_pi0[p] = logits - np.log(np.sum(np.exp(logits - logits.max()))) - (logits.max() - logits.max())
        # Numerically stable softmax normalisation
        log_pi0[p] = logits - _logsumexp(logits)

    return TaskDistribution(tasks=tasks, log_pi0=log_pi0, prior_strength=prior_strength)


def _logsumexp(logits: np.ndarray) -> float:
    """Numerically stable log-sum-exp of a 1-D array."""
    m = logits.max()
    return m + np.log(np.sum(np.exp(logits - m)))


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------

def reward_true(task: Task, action: int, rng=None) -> float:
    """Verifier-correct reward: 1 if action is correct, else 0."""
    return float(task.correct_mask[action])


def reward_random(task: Task, action: int, rng: np.random.Generator) -> float:
    """Bernoulli noise independent of correctness (p=0.5)."""
    return float(rng.integers(0, 2))


def reward_format(task: Task, action: int, rng=None) -> float:
    """Format reward: 1 if action == format_action, else 0.
    Correlated with prior (since prior also boosts format_action) but NOT
    with correctness (format_action is always incorrect by construction)."""
    return float(action == task.format_action)


def reward_frozen(task: Task, action: int, rng=None) -> float:
    """No-op baseline: always returns 0 (policy never updated)."""
    return 0.0


REWARD_FUNCTIONS = {
    "true":   reward_true,
    "random": reward_random,
    "format": reward_format,
    "frozen": reward_frozen,
}

# The "spurious" reward (self-consistency / majority vote, TTRL-style) is a
# GROUP-LEVEL signal: a completion's reward depends on the OTHER completions in
# its sampled group (does it match the group's plurality answer?). It therefore
# cannot be expressed as a per-action function and is handled directly inside
# ``grpo.run_grpo``. It is listed here for documentation / validation only.
GROUP_REWARD_TYPES = {"spurious"}
ALL_REWARD_TYPES = set(REWARD_FUNCTIONS) | GROUP_REWARD_TYPES


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_policy(
    log_pi: np.ndarray,
    tasks: list[Task],
    rng: np.random.Generator,
    n_samples: int = 64,
) -> float:
    """Estimate pass@1 accuracy of a policy by Monte Carlo.

    Parameters
    ----------
    log_pi : np.ndarray, shape (n_prompts, n_actions)
    tasks : list[Task]
    rng : np.random.Generator
    n_samples : int
        Number of i.i.d. samples per prompt.

    Returns
    -------
    float
        Mean pass@1 accuracy across prompts.
    """
    accs = []
    for p, task in enumerate(tasks):
        probs = np.exp(log_pi[p] - _logsumexp(log_pi[p]))
        samples = rng.choice(task.n_actions, size=n_samples, p=probs)
        accs.append(np.mean(task.correct_mask[samples]))
    return float(np.mean(accs))


def passk(
    log_pi: np.ndarray,
    tasks: list[Task],
    k: int = 4,
) -> float:
    """Exact pass@k (probability that at least one of k samples is correct).

    Uses the complement formula to avoid sampling.

    Returns
    -------
    float
        Mean pass@k across prompts.
    """
    accs = []
    for p, task in enumerate(tasks):
        probs = np.exp(log_pi[p] - _logsumexp(log_pi[p]))
        p_correct_single = np.sum(probs[task.correct_mask])
        p_all_wrong = (1.0 - p_correct_single) ** k
        accs.append(1.0 - p_all_wrong)
    return float(np.mean(accs))
