#!/usr/bin/env bash
# =============================================================================
# run_all_experiments.sh — LIGHT experiments for RLVR Reward Partition paper
#
# Runs all decisive experiments at modest scale.
# Expected wall time: < 4 minutes on a modern CPU.
#
# Outputs:
#   results/prior_sweep.json
#   results/factorial.json
#   results/points_vs_bounds.json
#   results/reaudit.json
#   results/power.json
#   ../paper/figures/fig_*.{pdf,png}
#   results/figures/fig_*.{pdf,png}
#
# Usage:
#   bash experiments/run_all_experiments.sh
#   bash experiments/run_all_experiments.sh --fast   (even fewer seeds)
# =============================================================================

set -euo pipefail

# cd to the experiments/ directory so relative imports work correctly
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Code root (parent of experiments/)
CODE_ROOT="$(dirname "$SCRIPT_DIR")"

# Ensure results/ and figures/ directories exist
mkdir -p "$CODE_ROOT/results/figures"
mkdir -p "$CODE_ROOT/../paper/figures"

# Python interpreter
PYTHON="${PYTHON:-python3}"

# -------------------------------------------------------------------
# Shared light settings
SEED=42
N_SEEDS=8
N_STEPS=200
N_PROMPTS=40
N_ACTIONS=8
G=8
LR=0.05
BETA=0.01
BATCH_SIZE=16
N_BOOTSTRAP=300

echo "======================================================"
echo " RLVR Reward Partition — Light Experiments"
echo " Seed=${SEED}  N_seeds=${N_SEEDS}  N_steps=${N_STEPS}"
echo "======================================================"
echo ""

# -------------------------------------------------------------------
echo "[1/5] Prior-strength sweep ..."
$PYTHON "$SCRIPT_DIR/exp_prior_sweep.py" \
    --seed $SEED \
    --n_seeds $N_SEEDS \
    --n_steps $N_STEPS \
    --n_prompts $N_PROMPTS \
    --n_actions $N_ACTIONS \
    --G $G \
    --lr $LR \
    --beta $BETA \
    --batch_size $BATCH_SIZE
echo ""

# -------------------------------------------------------------------
echo "[2/5] 2x2x2 Factorial ..."
$PYTHON "$SCRIPT_DIR/exp_factorial.py" \
    --seed $SEED \
    --n_seeds $N_SEEDS \
    --n_steps $N_STEPS \
    --n_prompts $N_PROMPTS \
    --n_actions $N_ACTIONS \
    --hi_prior 0.80 \
    --lo_prior 0.35 \
    --G $G \
    --lr $LR \
    --beta $BETA \
    --batch_size $BATCH_SIZE
echo ""

# -------------------------------------------------------------------
echo "[3/5] Points vs Bounds ..."
$PYTHON "$SCRIPT_DIR/exp_points_vs_bounds.py" \
    --seed $SEED \
    --n_seeds $N_SEEDS \
    --n_steps $N_STEPS \
    --n_prompts $N_PROMPTS \
    --n_actions $N_ACTIONS \
    --G $G \
    --lr $LR \
    --beta $BETA \
    --batch_size $BATCH_SIZE \
    --n_bootstrap $N_BOOTSTRAP
echo ""

# -------------------------------------------------------------------
echo "[4/5] Re-audit ..."
$PYTHON "$SCRIPT_DIR/exp_reaudit.py" \
    --seed $SEED \
    --n_seeds $N_SEEDS \
    --n_steps $N_STEPS \
    --n_prompts $N_PROMPTS \
    --n_actions $N_ACTIONS \
    --G $G \
    --lr $LR \
    --beta $BETA \
    --batch_size $BATCH_SIZE \
    --n_bootstrap $N_BOOTSTRAP
echo ""

# -------------------------------------------------------------------
echo "[5/5] Power analysis ..."
$PYTHON "$SCRIPT_DIR/exp_power.py" \
    --seed $SEED \
    --n_steps $N_STEPS \
    --n_prompts $N_PROMPTS \
    --n_actions $N_ACTIONS \
    --G $G \
    --lr $LR \
    --beta $BETA \
    --batch_size $BATCH_SIZE \
    --n_bootstrap $N_BOOTSTRAP \
    --prior_strength 0.50
echo ""

# -------------------------------------------------------------------
echo "======================================================"
echo " All experiments complete."
echo " Results in: $CODE_ROOT/results/"
echo " Figures in: $CODE_ROOT/../paper/figures/"
echo "======================================================"

# Print summary of results
echo ""
echo "--- JSON results ---"
for f in "$CODE_ROOT/results/"*.json; do
    echo "  $f"
done
echo ""
echo "--- Figures ---"
for f in "$CODE_ROOT/../paper/figures/"*.pdf; do
    echo "  $f"
done
