#!/usr/bin/env bash
# Launch the full real-GRPO matrix across GPUs (one (model,reward) per GPU).
# Strong-prior family: Qwen2.5-1.5B-Instruct ; weak-prior family: Llama-3.2-1B-Instruct.
# Rewards: frozen (baseline), random, spurious (self-consistency), true (verifier).
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
OUT="${OUT:-results_grpo}"
mkdir -p "$OUT" logs
STEPS=${STEPS:-80}; B=${B:-6}; G=${G:-6}; MB=${MB:-4}; MAXNEW=${MAXNEW:-200}; NEVAL=${NEVAL:-50}; EVERY=${EVERY:-20}

QWEN="Qwen/Qwen2.5-1.5B-Instruct"
LLAMA="meta-llama/Llama-3.2-1B-Instruct"

# (gpu, model, reward)
run () {
  local gpu=$1 model=$2 reward=$3
  local tag="$(basename $model)_$reward"
  echo "GPU $gpu -> $tag"
  CUDA_VISIBLE_DEVICES=$gpu nohup python3 gpu_grpo.py --model "$model" --reward "$reward" \
    --steps $STEPS --B $B --G $G --micro_bs $MB --max_new $MAXNEW --n_eval $NEVAL --eval_every $EVERY \
    --out "$OUT" > "logs/${tag}.log" 2>&1 &
}

# 6 training runs on GPUs 1-6
run 1 "$QWEN"  true
run 2 "$QWEN"  random
run 3 "$QWEN"  spurious
run 4 "$LLAMA" true
run 5 "$LLAMA" random
run 6 "$LLAMA" spurious
# 2 frozen baselines on GPU 7 (cheap, run sequentially after a short delay)
( CUDA_VISIBLE_DEVICES=7 python3 gpu_grpo.py --model "$QWEN"  --reward frozen --n_eval $NEVAL --max_new $MAXNEW --out "$OUT" > logs/$(basename $QWEN)_frozen.log 2>&1
  CUDA_VISIBLE_DEVICES=7 python3 gpu_grpo.py --model "$LLAMA" --reward frozen --n_eval $NEVAL --max_new $MAXNEW --out "$OUT" > logs/$(basename $LLAMA)_frozen.log 2>&1 ) &

echo "all launched. monitor with: ls $OUT/*.json ; tail -f logs/*.log"
wait
echo "ALL GRPO RUNS COMPLETE"
