#!/usr/bin/env bash
# =============================================================================
# run_mess_compute_all.sh — HEAVY experiments for RLVR Reward Partition paper
#
# Runs large-N sweeps, many bootstrap reps, and an optional real-model
# Best-of-N proxy on gpt2/distilgpt2 (skipped if transformers unavailable).
#
# Expected wall time: 30–120 minutes (CPU-only).
#                     Real RLVR on 0.5–3B needs a GPU — recipe in comments.
#
# Usage:
#   bash experiments/run_mess_compute_all.sh
#   bash experiments/run_mess_compute_all.sh --skip-bon   (skip real model)
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$SCRIPT_DIR"

mkdir -p "$CODE_ROOT/results/figures"
mkdir -p "$CODE_ROOT/../paper/figures"

PYTHON="${PYTHON:-python3}"
SKIP_BON="${1:-}"

echo "======================================================"
echo " RLVR Reward Partition — Heavy Experiments"
echo "======================================================"
echo ""

# -------------------------------------------------------------------
# (a) Large factorial: 100+ seeds per cell, 500 steps
echo "[a] Large 2x2x2 factorial (100 seeds, 500 steps) ..."
$PYTHON "$SCRIPT_DIR/exp_factorial.py" \
    --seed 0 \
    --n_seeds 100 \
    --n_steps 500 \
    --n_prompts 80 \
    --n_actions 10 \
    --hi_prior 0.80 \
    --lo_prior 0.35 \
    --G 10 \
    --lr 0.05 \
    --beta 0.01 \
    --batch_size 20
# rename output so it doesn't overwrite light results
cp "$CODE_ROOT/results/factorial.json" "$CODE_ROOT/results/factorial_heavy.json"
echo "  saved factorial_heavy.json"
echo ""

# -------------------------------------------------------------------
# (b) Large prior sweep with denser grid
echo "[b] Dense prior sweep ..."
$PYTHON "$SCRIPT_DIR/exp_prior_sweep.py" \
    --seed 0 \
    --n_seeds 50 \
    --n_steps 400 \
    --n_prompts 80 \
    --n_actions 10 \
    --G 10 \
    --lr 0.05 \
    --beta 0.01 \
    --batch_size 20
cp "$CODE_ROOT/results/prior_sweep.json" "$CODE_ROOT/results/prior_sweep_heavy.json"
echo ""

# -------------------------------------------------------------------
# (c) Optional real Best-of-N proxy on gpt2/distilgpt2
#
# NOTE: This is a Best-of-N experiment, NOT real RLVR.  It illustrates
# the "spurious rewards surface latent prior" phenomenon in a different
# setting: we sample N completions from a frozen language model and keep
# the one with the highest reward (true or random).  With random rewards
# we still get gains because BoN with N>1 already exploits the model's
# latent prior.  This is clearly labelled as BoN in all outputs.
#
# Skipped if: (1) --skip-bon flag, (2) transformers not importable,
#             (3) gpt2 model files not available.
#
# For real RLVR on 0.5B–3B models (needs GPU):
#   1. Install: pip install torch transformers trl accelerate
#   2. Choose a model: Qwen2.5-Math-1.5B or deepseek-math-7b-instruct
#   3. Run GRPO via TRL: trl/examples/research_projects/stack_llama/grpo_trainer.py
#   4. Reward types: verifier (Math-Shepherd or custom), random Bernoulli
#   5. Log acc@1 on MATH-500 every 50 steps for 2000 steps
#   6. Reproduce Figure 1 from the paper with real curves
if [[ "$SKIP_BON" == "--skip-bon" ]]; then
    echo "[c] Skipping BoN proxy (--skip-bon flag)."
else
    echo "[c] Optional BoN proxy on gpt2 ..."
    $PYTHON "$SCRIPT_DIR/exp_bon_proxy.py" || echo "  [SKIPPED] BoN proxy failed (transformers/model unavailable)."
fi
echo ""

# ----------------------------------------------------------------------
# (d) REAL-MODEL validation: Llama-3.2-1B best-of-N on GSM8K (EXECUTED).
# This is the experiment reported in the paper (fig_realmodel_bon).
# Requires: pip install torch transformers datasets ; cached or
# downloadable meta-llama/Llama-3.2-1B-Instruct + openai/gsm8k.
# ~20 min CPU (24 problems x N=6 CoT samples). Override with flags:
#   --model <hf_id>  --n_problems <P>  --N <k>  --max_new_tokens <t>
# ----------------------------------------------------------------------
if [[ "$SKIP_BON" == "--skip-bon" ]]; then
    echo "[d] Skipping real-model GSM8K validation (--skip-bon flag)."
else
    echo "[d] Real-model best-of-N validation (Llama-3.2-1B on GSM8K) ..."
    $PYTHON "$SCRIPT_DIR/exp_realmodel_bon.py" --n_problems 24 --N 6 --seed 42 --max_new_tokens 170 \
        || echo "  [SKIPPED] real-model validation failed (torch/model/dataset unavailable)."
fi
echo ""

echo "======================================================"
echo " Heavy experiments complete."
echo "======================================================"

# ----------------------------------------------------------------------
# Real GRPO RLVR (GPU): LoRA-GRPO on Qwen2.5-1.5B + Llama-3.2-1B over GSM8K
# under {frozen,random,spurious,true} rewards, then aggregate the partition.
# Requires: pip install torch transformers peft datasets accelerate ; GPUs.
# Launches one (model,reward) per GPU; ~2h. See gpu/ for the trainer.
# ----------------------------------------------------------------------
echo "[real-RLVR] launching GRPO matrix on available GPUs ..."
( cd gpu && bash gpu_grpo_launch.sh && python3 aggregate_grpo.py ) \
    || echo "  [SKIPPED] needs GPUs + torch/peft for real GRPO"
