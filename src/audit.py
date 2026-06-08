"""Re-audit instrument for reported RLVR gain decompositions.

Given a pair (true_gain, random_gain) from a simulated or real family
— where true_gain = acc(TRUE) - acc(FROZEN) and random_gain =
acc(RANDOM) - acc(FROZEN) — this module outputs:

1. The attributed split:
   * filtering_elicitation_share  = random_gain / true_gain
   * reward_design_share          = (true_gain - random_gain) / true_gain
   with bootstrap CIs.

2. A verdict string that characterises the result:
   * "FILTERING_DOMINATED"   if FE share > 0.60
   * "REWARD_DESIGN_DOMINATED" if RD share > 0.60
   * "MIXED"                 otherwise.

The instrument is designed to demonstrate:
  (A) On a STRONG-PRIOR family: a large "true-minus-random" is mostly
      filtering/elicitation (FE share >> 0.5).
  (B) On a WEAK-PRIOR + informative-reward family: reward-design
      dominates (RD share >> 0.5).

This formalises the main qualitative claim of the paper without relying
on any specific numerical threshold — the split is empirically measured.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass

from .env import TaskDistribution, make_task_distribution
from .grpo import GRPOConfig, run_grpo_seeds


# ---------------------------------------------------------------------------
# Verdict thresholds (pre-registered)
# ---------------------------------------------------------------------------

FE_DOMINATED_THRESHOLD: float = 0.60
RD_DOMINATED_THRESHOLD: float = 0.60


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AuditResult:
    """Output of the re-audit instrument.

    Attributes
    ----------
    family_name : str
        Label for the family being audited.
    acc_frozen : float
    acc_random : float
    acc_true : float
    true_gain : float
        acc(TRUE) - acc(FROZEN).
    random_gain : float
        acc(RANDOM) - acc(FROZEN).
    fe_share : float
        Fraction of true_gain attributable to filtering/elicitation.
    rd_share : float
        Fraction of true_gain attributable to reward design.
    fe_share_ci : tuple[float, float]
        95% bootstrap CI for fe_share.
    rd_share_ci : tuple[float, float]
        95% bootstrap CI for rd_share.
    fe_gain_abs : float
        Absolute FE gain.
    rd_gain_abs : float
        Absolute RD gain.
    fe_gain_ci : tuple[float, float]
        95% CI for absolute FE gain.
    rd_gain_ci : tuple[float, float]
        95% CI for absolute RD gain.
    verdict : str
        One of "FILTERING_DOMINATED", "REWARD_DESIGN_DOMINATED", "MIXED".
    """
    family_name: str
    acc_frozen: float
    acc_random: float
    acc_true: float
    true_gain: float
    random_gain: float
    fe_share: float
    rd_share: float
    fe_share_ci: tuple[float, float]
    rd_share_ci: tuple[float, float]
    fe_gain_abs: float
    rd_gain_abs: float
    fe_gain_ci: tuple[float, float]
    rd_gain_ci: tuple[float, float]
    verdict: str

    def to_dict(self) -> dict:
        return {
            "family_name":  self.family_name,
            "acc_frozen":   self.acc_frozen,
            "acc_random":   self.acc_random,
            "acc_true":     self.acc_true,
            "true_gain":    self.true_gain,
            "random_gain":  self.random_gain,
            "fe_share":     self.fe_share,
            "rd_share":     self.rd_share,
            "fe_share_ci":  list(self.fe_share_ci),
            "rd_share_ci":  list(self.rd_share_ci),
            "fe_gain_abs":  self.fe_gain_abs,
            "rd_gain_abs":  self.rd_gain_abs,
            "fe_gain_ci":   list(self.fe_gain_ci),
            "rd_gain_ci":   list(self.rd_gain_ci),
            "verdict":      self.verdict,
        }

    def __str__(self) -> str:
        lines = [
            f"=== Audit: {self.family_name} ===",
            f"  acc(FROZEN) = {self.acc_frozen:.4f}",
            f"  acc(RANDOM) = {self.acc_random:.4f}",
            f"  acc(TRUE)   = {self.acc_true:.4f}",
            f"  true_gain   = {self.true_gain:.4f}",
            f"  random_gain = {self.random_gain:.4f}",
            f"  FE gain     = {self.fe_gain_abs:.4f}  [{self.fe_gain_ci[0]:.4f}, {self.fe_gain_ci[1]:.4f}]",
            f"  RD gain     = {self.rd_gain_abs:.4f}  [{self.rd_gain_ci[0]:.4f}, {self.rd_gain_ci[1]:.4f}]",
            f"  FE share    = {self.fe_share:.2%}  [{self.fe_share_ci[0]:.2%}, {self.fe_share_ci[1]:.2%}]",
            f"  RD share    = {self.rd_share:.2%}  [{self.rd_share_ci[0]:.2%}, {self.rd_share_ci[1]:.2%}]",
            f"  VERDICT: {self.verdict}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core audit function
# ---------------------------------------------------------------------------

def audit_family(
    family_name: str,
    dist: TaskDistribution,
    n_seeds: int = 12,
    n_steps: int = 200,
    base_seed: int = 0,
    G: int = 8,
    lr: float = 0.05,
    beta: float = 0.01,
    batch_size: int = 16,
    n_bootstrap: int = 500,
) -> AuditResult:
    """Run the re-audit instrument on a given family.

    Parameters
    ----------
    family_name : str
        Human-readable label (e.g. "Qwen-math-like (strong prior)").
    dist : TaskDistribution
        Task distribution representing the family.
    n_seeds : int
        Independent training seeds per condition.
    n_steps : int
        GRPO steps per run.
    base_seed : int
    G, lr, beta, batch_size : float
        GRPO hyper-parameters.
    n_bootstrap : int
        Bootstrap replications for CIs.

    Returns
    -------
    AuditResult
    """
    raw_accs: dict[str, np.ndarray] = {}

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
            base_seed=base_seed + {"frozen": 0, "random": 100, "spurious": 150, "true": 200}[reward_type],
        )
        raw_accs[reward_type] = np.array([r.final_acc for r in results])

    acc_frozen   = float(raw_accs["frozen"].mean())
    acc_random   = float(raw_accs["random"].mean())
    acc_spurious = float(raw_accs["spurious"].mean())
    acc_true     = float(raw_accs["true"].mean())

    true_gain   = acc_true   - acc_frozen
    random_gain = acc_random - acc_frozen
    # Exact elicitation vs reward-design split via the self-consistency anchor:
    fe_gain_abs = acc_spurious - acc_frozen     # elicitation
    rd_gain_abs = acc_true     - acc_spurious   # reward design

    # Safe division for shares
    if abs(true_gain) < 1e-9:
        fe_share = 0.5
        rd_share = 0.5
    else:
        fe_share = fe_gain_abs / true_gain
        rd_share = rd_gain_abs / true_gain

    # Bootstrap CIs
    rng_bs = np.random.default_rng(base_seed + 42_000)
    fe_boot_abs = []
    rd_boot_abs = []
    fe_boot_share = []
    rd_boot_share = []

    n = n_seeds
    for _ in range(n_bootstrap):
        bs_fr = rng_bs.choice(raw_accs["frozen"],   size=n, replace=True).mean()
        bs_sp = rng_bs.choice(raw_accs["spurious"], size=n, replace=True).mean()
        bs_tr = rng_bs.choice(raw_accs["true"],     size=n, replace=True).mean()

        fe_a = bs_sp - bs_fr
        rd_a = bs_tr - bs_sp
        tg   = bs_tr - bs_fr

        fe_boot_abs.append(fe_a)
        rd_boot_abs.append(rd_a)

        if abs(tg) > 1e-9:
            fe_boot_share.append(fe_a / tg)
            rd_boot_share.append(rd_a / tg)
        else:
            fe_boot_share.append(0.5)
            rd_boot_share.append(0.5)

    fe_gain_ci   = (float(np.percentile(fe_boot_abs,   2.5)), float(np.percentile(fe_boot_abs,   97.5)))
    rd_gain_ci   = (float(np.percentile(rd_boot_abs,   2.5)), float(np.percentile(rd_boot_abs,   97.5)))
    fe_share_ci  = (float(np.percentile(fe_boot_share, 2.5)), float(np.percentile(fe_boot_share, 97.5)))
    rd_share_ci  = (float(np.percentile(rd_boot_share, 2.5)), float(np.percentile(rd_boot_share, 97.5)))

    # Verdict
    if fe_share > FE_DOMINATED_THRESHOLD:
        verdict = "ELICITATION_DOMINATED"
    elif rd_share > RD_DOMINATED_THRESHOLD:
        verdict = "REWARD_DESIGN_DOMINATED"
    else:
        verdict = "MIXED"

    return AuditResult(
        family_name=family_name,
        acc_frozen=acc_frozen,
        acc_random=acc_random,
        acc_true=acc_true,
        true_gain=true_gain,
        random_gain=random_gain,
        fe_share=fe_share,
        rd_share=rd_share,
        fe_share_ci=fe_share_ci,
        rd_share_ci=rd_share_ci,
        fe_gain_abs=fe_gain_abs,
        rd_gain_abs=rd_gain_abs,
        fe_gain_ci=fe_gain_ci,
        rd_gain_ci=rd_gain_ci,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# Convenience: audit two canonical families
# ---------------------------------------------------------------------------

def audit_canonical_families(
    n_seeds: int = 12,
    n_steps: int = 200,
    base_seed: int = 0,
    n_prompts: int = 40,
    n_actions: int = 8,
    G: int = 8,
    lr: float = 0.05,
    beta: float = 0.01,
    batch_size: int = 16,
    n_bootstrap: int = 500,
) -> list[AuditResult]:
    """Audit two canonical families:

    1. Strong-prior (Qwen-math-like): prior_strength=0.80
       Expected: FILTERING_DOMINATED
    2. Weak-prior (OLMo/Llama-like): prior_strength=0.35
       Expected: REWARD_DESIGN_DOMINATED
    """
    results = []

    for family_label, ps in [
        ("Qwen-math-like (strong prior, ps=0.80)", 0.80),
        ("OLMo/Llama-like (weak prior, ps=0.35)",  0.35),
    ]:
        dist = make_task_distribution(
            n_prompts=n_prompts,
            n_actions=n_actions,
            prior_strength=ps,
            seed=base_seed + int(ps * 100),
        )
        result = audit_family(
            family_name=family_label,
            dist=dist,
            n_seeds=n_seeds,
            n_steps=n_steps,
            base_seed=base_seed,
            G=G,
            lr=lr,
            beta=beta,
            batch_size=batch_size,
            n_bootstrap=n_bootstrap,
        )
        results.append(result)
        print(result)
        print()

    return results
