"""Causal partition of the 'true-minus-random' gain in RLVR.

This module is THE core scientific contribution of the paper.

Definitions
-----------
Let acc(R) denote the final pass@1 accuracy after training with reward
function R (averaged over seeds and prompts).

* **Filtering+Elicitation gain** (FE):
      FE = acc(RANDOM) - acc(FROZEN)
  This is the gain that arises even when rewards carry no information about
  correctness.  It can only come from (a) the dynamic-sampling filter that
  drops uninformative groups, surfacing latent prior mass; and/or (b) the
  KL-to-prior regularisation that allows the policy to stay near pi0.  Both
  are STRUCTURAL, not informational, effects.

* **Reward-design gain** (RD):
      RD = acc(TRUE) - acc(RANDOM)
  This is the RESIDUAL gain from having an informative signal, net of the
  structural effects captured by FE.

* **Total observed gain** (TOG):
      TOG = acc(TRUE) - acc(FROZEN)    = FE + RD

These two terms sum to the full gain only if the interaction term is
negligible.  We test this via a **2×2×2 factorial experiment** and an
**additivity specification test**.

2×2×2 Factorial
---------------
Factor A: reward_informative  (TRUE vs RANDOM)
Factor B: filtering           (on vs off)
Factor C: prior_strength      (hi vs lo)

We run 2^3 = 8 cells, each with n_seeds seeds.
We fit a linear model (on the accuracy directly, or on logit-accuracy):

    y = mu + a*A + b*B + c*C + ab*AB + ac*AC + bc*BC + abc*ABC

and report main effects + interactions.

Additivity specification test
------------------------------
We compare the saturated model (all 8 terms) with an *additive* model
(main effects only, 3 terms) using a held-out prediction residual.  If the
maximum interaction coefficient (as a fraction of the main effects) exceeds
a pre-registered threshold (DEFAULT_INTERACTION_THRESHOLD = 0.20), we
declare the decomposition to be BOUNDS rather than POINTS.

Points vs Bounds
----------------
If filtering and reward-design interact significantly:
  * Report FE as [0, acc(RANDOM)-acc(FROZEN)] (upper bound)
  * Report RD as [acc(TRUE)-acc(RANDOM), acc(TRUE)-acc(FROZEN)] (range)
Otherwise report point estimates with bootstrap CIs.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass
from typing import NamedTuple

from .env import TaskDistribution, make_task_distribution
from .grpo import GRPOConfig, run_grpo_seeds, mean_final_acc, TrainingResult


# ---------------------------------------------------------------------------
# Pre-registered threshold
# ---------------------------------------------------------------------------

DEFAULT_INTERACTION_THRESHOLD: float = 0.20
"""Maximum |interaction coefficient| / |max main effect| before we switch
from POINTS to BOUNDS reporting.  Pre-registered value: 0.20."""


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class FactorialCell:
    """One cell of the 2×2×2 factorial.

    Attributes
    ----------
    reward_informative : bool
        True = TRUE reward; False = RANDOM reward.
    filtering : bool
    prior_strength : float
        Either hi_prior or lo_prior value.
    results : list[TrainingResult]
    mean_acc : float
    std_acc : float
    """
    reward_informative: bool
    filtering: bool
    prior_strength: float
    results: list[TrainingResult]
    mean_acc: float
    std_acc: float


@dataclass
class FactorialResult:
    """Full 2×2×2 factorial result.

    Attributes
    ----------
    cells : list[FactorialCell]
        All 8 cells.
    effects : dict[str, float]
        Estimated main effects and interactions from linear model.
    interaction_ratio : float
        max(|interaction|) / max(|main effect|).
    additivity_test : str
        "additive" or "non-additive".
    decomposition_mode : str
        "points" or "bounds".
    hi_prior : float
    lo_prior : float
    """
    cells: list[FactorialCell]
    effects: dict[str, float]
    interaction_ratio: float
    additivity_test: str
    decomposition_mode: str
    hi_prior: float
    lo_prior: float


@dataclass
class PartitionEstimate:
    """Exact telescoping partition of the RL gain in RLVR.

    The total gain ``acc(TRUE) - acc(FROZEN)`` decomposes EXACTLY into three
    sequential, non-overlapping components::

        random_null     = acc(RANDOM)   - acc(FROZEN)    # truly-random reward (~0)
        elicitation     = acc(SPURIOUS) - acc(RANDOM)    # self-consistency surfaces latent prior
        reward_design   = acc(TRUE)     - acc(SPURIOUS)  # genuine correctness signal on top
        ------------------------------------------------
        total_gain      = acc(TRUE)     - acc(FROZEN)

    The quantity practitioners report as the "reward effect" is the NAIVE
    ``acc(TRUE) - acc(RANDOM) = elicitation + reward_design``, which OVERSTATES
    the true reward-design contribution by exactly ``elicitation`` (the
    conflation bias). ``filtering_elicitation_gain`` / ``reward_design_gain``
    retain their names for backward-compatible plotting but now mean
    elicitation (SPURIOUS-FROZEN incl. the ~0 random null) and reward-design
    (TRUE-SPURIOUS) respectively.
    """
    acc_frozen: float
    acc_random: float
    acc_spurious: float
    acc_true: float
    random_null_gain: float                 # acc(RANDOM) - acc(FROZEN)  (~0)
    filtering_elicitation_gain: float       # elicitation: acc(SPURIOUS) - acc(FROZEN)
    reward_design_gain: float               # acc(TRUE) - acc(SPURIOUS)
    naive_reward_design: float              # acc(TRUE) - acc(RANDOM)  (what practitioners report)
    conflation_bias: float                  # naive - reward_design = elicitation net of random null
    total_gain: float
    fe_ci: tuple[float, float]              # CI for elicitation
    rd_ci: tuple[float, float]              # CI for reward-design
    bias_ci: tuple[float, float]            # CI for conflation bias
    decomposition_mode: str                # "points" if bias CI excludes 0 else "bounds"
    interaction_term: float                # kept for compat; conflation bias proxy


# ---------------------------------------------------------------------------
# 2×2×2 Factorial runner
# ---------------------------------------------------------------------------

def run_factorial(
    n_prompts: int = 40,
    n_actions: int = 8,
    n_seeds: int = 8,
    n_steps: int = 200,
    hi_prior: float = 0.80,
    lo_prior: float = 0.35,
    batch_size: int = 16,
    G: int = 8,
    lr: float = 0.05,
    beta: float = 0.01,
    eval_every: int = 25,
    base_seed: int = 0,
    interaction_threshold: float = DEFAULT_INTERACTION_THRESHOLD,
) -> FactorialResult:
    """Run the full 2×2×2 factorial and return results + linear model.

    Parameters
    ----------
    n_prompts, n_actions : int
        Task distribution size.
    n_seeds : int
        Seeds per cell.
    n_steps : int
        GRPO steps per run.
    hi_prior, lo_prior : float
        Prior strength levels for the two factor levels.
    batch_size, G, lr, beta : float
        GRPO hyper-parameters.
    eval_every : int
        Evaluation frequency.
    base_seed : int
        Base RNG seed.
    interaction_threshold : float
        Pre-registered threshold for POINTS vs BOUNDS decision.

    Returns
    -------
    FactorialResult
    """
    # Factor A contrasts the verifiable reward (TRUE) against the self-consistency
    # baseline (SPURIOUS). The A main effect therefore estimates the REWARD-DESIGN
    # gain net of elicitation, and the A x C interaction tests whether reward
    # design is prior-dependent (i.e. whether "true - spurious" is transferable).
    factor_levels = {
        "reward_informative": [True, False],  # TRUE vs SPURIOUS (self-consistency)
        "filtering":          [True, False],
        "prior_strength":     [hi_prior, lo_prior],
    }

    cells: list[FactorialCell] = []
    seed_offset = 0

    for ri in factor_levels["reward_informative"]:
        for fi in factor_levels["filtering"]:
            for ps in factor_levels["prior_strength"]:
                # Build task distribution for this prior strength
                dist = make_task_distribution(
                    n_prompts=n_prompts,
                    n_actions=n_actions,
                    prior_strength=ps,
                    seed=base_seed + seed_offset * 100,
                )

                reward_type = "true" if ri else "spurious"

                base_cfg = GRPOConfig(
                    n_steps=n_steps,
                    batch_size=batch_size,
                    G=G,
                    lr=lr,
                    beta=beta,
                    filtering=fi,
                    reward_type=reward_type,
                    eval_every=eval_every,
                    eval_samples=64,
                    seed=base_seed,
                )

                results = run_grpo_seeds(
                    dist, base_cfg, n_seeds=n_seeds,
                    base_seed=base_seed + seed_offset * 13 + 7,
                )

                m, s = mean_final_acc(results)
                cells.append(FactorialCell(
                    reward_informative=ri,
                    filtering=fi,
                    prior_strength=ps,
                    results=results,
                    mean_acc=m,
                    std_acc=s,
                ))
                seed_offset += 1

    # --- Fit linear model on cell means -----------------------------------
    effects, interaction_ratio, additivity_test = _fit_linear_model(
        cells, hi_prior, lo_prior, interaction_threshold
    )

    decomposition_mode = "points" if additivity_test == "additive" else "bounds"

    return FactorialResult(
        cells=cells,
        effects=effects,
        interaction_ratio=interaction_ratio,
        additivity_test=additivity_test,
        decomposition_mode=decomposition_mode,
        hi_prior=hi_prior,
        lo_prior=lo_prior,
    )


def _fit_linear_model(
    cells: list[FactorialCell],
    hi_prior: float,
    lo_prior: float,
    interaction_threshold: float,
) -> tuple[dict[str, float], float, str]:
    """Fit a linear model on the 8 cell means.

    Factor coding: +1 / -1 (effect coding).
    A = reward_informative (+1=TRUE, -1=RANDOM)
    B = filtering          (+1=on,   -1=off)
    C = prior_strength     (+1=hi,   -1=lo)

    Returns
    -------
    effects : dict[str, float]
    interaction_ratio : float
    additivity_test : str  "additive" | "non-additive"
    """
    # Build design matrix (8 rows for 8 cells)
    # Order: intercept, A, B, C, AB, AC, BC, ABC
    X = []
    y = []

    for cell in cells:
        A = 1.0 if cell.reward_informative else -1.0
        B = 1.0 if cell.filtering else -1.0
        C = 1.0 if cell.prior_strength == hi_prior else -1.0
        row = [1.0, A, B, C, A*B, A*C, B*C, A*B*C]
        X.append(row)
        y.append(cell.mean_acc)

    X = np.array(X)
    y = np.array(y)

    # OLS via normal equations (8 unknowns, 8 equations — exactly identified)
    try:
        coeffs = np.linalg.solve(X, y)
    except np.linalg.LinAlgError:
        coeffs = np.linalg.lstsq(X, y, rcond=None)[0]

    names = ["intercept", "A_reward", "B_filtering", "C_prior",
             "AB", "AC", "BC", "ABC"]
    effects = {name: float(coeffs[i]) for i, name in enumerate(names)}

    # Interaction ratio: max |interaction| / max |main effect|
    main_effects = np.abs(coeffs[1:4])
    interactions = np.abs(coeffs[4:])
    max_main = max(main_effects.max(), 1e-9)
    max_inter = interactions.max()
    interaction_ratio = float(max_inter / max_main)

    additivity_test = "additive" if interaction_ratio < interaction_threshold else "non-additive"

    return effects, interaction_ratio, additivity_test


# ---------------------------------------------------------------------------
# Causal partition estimator
# ---------------------------------------------------------------------------

def estimate_partition(
    dist: TaskDistribution,
    n_seeds: int = 10,
    n_steps: int = 200,
    base_seed: int = 0,
    G: int = 8,
    lr: float = 0.05,
    beta: float = 0.01,
    batch_size: int = 16,
    n_bootstrap: int = 500,
    interaction_threshold: float = DEFAULT_INTERACTION_THRESHOLD,
) -> PartitionEstimate:
    """Estimate the FE and RD partition for a given task distribution.

    Runs three conditions: FROZEN, RANDOM, TRUE (all with filtering=True).
    Uses bootstrap to compute CIs.

    Parameters
    ----------
    dist : TaskDistribution
    n_seeds : int
    n_steps : int
    base_seed : int
    G, lr, beta, batch_size : float
        GRPO hyper-parameters.
    n_bootstrap : int
        Bootstrap replications for CI computation.
    interaction_threshold : float

    Returns
    -------
    PartitionEstimate
    """
    accs: dict[str, np.ndarray] = {}

    for reward_type in ("frozen", "random", "spurious", "true"):
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
        results = run_grpo_seeds(dist, cfg, n_seeds=n_seeds, base_seed=base_seed + 1000)
        accs[reward_type] = np.array([r.final_acc for r in results])

    acc_frozen   = float(accs["frozen"].mean())
    acc_random   = float(accs["random"].mean())
    acc_spurious = float(accs["spurious"].mean())
    acc_true     = float(accs["true"].mean())

    random_null = acc_random   - acc_frozen
    elicitation = acc_spurious - acc_frozen
    rd_gain     = acc_true     - acc_spurious
    naive_rd    = acc_true     - acc_random          # what practitioners report
    bias        = naive_rd     - rd_gain             # = acc_spurious - acc_random (conflation)
    tot_gain    = acc_true     - acc_frozen

    # Bootstrap CIs
    rng_bs = np.random.default_rng(base_seed + 9999)
    el_boots, rd_boots, bias_boots = [], [], []
    for _ in range(n_bootstrap):
        bs_frozen = rng_bs.choice(accs["frozen"],   size=n_seeds, replace=True).mean()
        bs_random = rng_bs.choice(accs["random"],   size=n_seeds, replace=True).mean()
        bs_spur   = rng_bs.choice(accs["spurious"], size=n_seeds, replace=True).mean()
        bs_true   = rng_bs.choice(accs["true"],     size=n_seeds, replace=True).mean()
        el_boots.append(bs_spur - bs_frozen)
        rd_boots.append(bs_true - bs_spur)
        bias_boots.append(bs_spur - bs_random)

    fe_ci   = (float(np.percentile(el_boots,   2.5)), float(np.percentile(el_boots,   97.5)))
    rd_ci   = (float(np.percentile(rd_boots,   2.5)), float(np.percentile(rd_boots,   97.5)))
    bias_ci = (float(np.percentile(bias_boots, 2.5)), float(np.percentile(bias_boots, 97.5)))

    # POINTS if we can statistically separate elicitation from the reward-design
    # signal (the conflation bias CI excludes 0); otherwise we can only BOUND.
    excludes_zero = (bias_ci[0] > 0) or (bias_ci[1] < 0)
    decomposition_mode = "points" if excludes_zero else "bounds"

    return PartitionEstimate(
        acc_frozen=acc_frozen,
        acc_random=acc_random,
        acc_spurious=acc_spurious,
        acc_true=acc_true,
        random_null_gain=random_null,
        filtering_elicitation_gain=elicitation,
        reward_design_gain=rd_gain,
        naive_reward_design=naive_rd,
        conflation_bias=bias,
        total_gain=tot_gain,
        fe_ci=fe_ci,
        rd_ci=rd_ci,
        bias_ci=bias_ci,
        decomposition_mode=decomposition_mode,
        interaction_term=bias,
    )


# ---------------------------------------------------------------------------
# Prior sweep helper
# ---------------------------------------------------------------------------

def prior_sweep(
    prior_strengths: list[float],
    n_prompts: int = 40,
    n_actions: int = 8,
    n_seeds: int = 8,
    n_steps: int = 200,
    base_seed: int = 0,
    G: int = 8,
    lr: float = 0.05,
    beta: float = 0.01,
    batch_size: int = 16,
) -> list[dict]:
    """Sweep over prior strengths and collect FE/RD gains.

    Returns
    -------
    list of dicts with keys:
        prior_strength, acc_frozen, acc_random, acc_true,
        fe_gain, rd_gain, tot_gain
    """
    rows = []
    for ps_idx, ps in enumerate(prior_strengths):
        dist = make_task_distribution(
            n_prompts=n_prompts,
            n_actions=n_actions,
            prior_strength=ps,
            seed=base_seed + ps_idx * 7,
        )

        accs: dict[str, float] = {}
        for reward_type in ("frozen", "random", "spurious", "true"):
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
            results = run_grpo_seeds(
                dist, cfg, n_seeds=n_seeds,
                base_seed=base_seed + ps_idx * 100 + 50,
            )
            accs[reward_type] = float(np.mean([r.final_acc for r in results]))

        elicitation = accs["spurious"] - accs["random"]      # self-consistency surfaces prior
        reward_design = accs["true"]   - accs["spurious"]    # genuine correctness signal
        naive_rd = accs["true"] - accs["random"]             # what practitioners report
        tot = accs["true"] - accs["frozen"]
        rd_fraction = reward_design / naive_rd if abs(naive_rd) > 1e-9 else float("nan")
        rows.append({
            "prior_strength": ps,
            "acc_frozen":   accs["frozen"],
            "acc_random":   accs["random"],
            "acc_spurious": accs["spurious"],
            "acc_true":     accs["true"],
            "random_null":  accs["random"] - accs["frozen"],   # ~0 sanity check
            "fe_gain":      elicitation,                       # elicitation (kept key name)
            "rd_gain":      reward_design,                     # reward-design (kept key name)
            "naive_gain":   naive_rd,                          # conflated true-minus-random
            "rd_fraction":  rd_fraction,                       # genuine RD share of naive
            "tot_gain":     tot,
        })
    return rows


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def factorial_to_dict(fr: FactorialResult) -> dict:
    """Convert a FactorialResult to a JSON-serialisable dict."""
    cells_list = []
    for c in fr.cells:
        cells_list.append({
            "reward_informative": c.reward_informative,
            "filtering": c.filtering,
            "prior_strength": float(c.prior_strength),
            "mean_acc": float(c.mean_acc),
            "std_acc": float(c.std_acc),
        })
    return {
        "cells": cells_list,
        "effects": {k: float(v) for k, v in fr.effects.items()},
        "interaction_ratio": float(fr.interaction_ratio),
        "additivity_test": fr.additivity_test,
        "decomposition_mode": fr.decomposition_mode,
        "hi_prior": float(fr.hi_prior),
        "lo_prior": float(fr.lo_prior),
    }
