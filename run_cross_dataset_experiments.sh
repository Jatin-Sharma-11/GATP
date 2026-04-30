#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

GPU_LIST="${1:-${GPU_LIST:-${GPU:-0}}}"
PRETRAIN_ONLY="${PRETRAIN_ONLY:-0}"
MAX_JOBS_PER_GPU="${MAX_JOBS_PER_GPU:-1}"

pairs=(
  "PEMS03 PEMS04"
  "PEMS03 PEMS07"
  "PEMS03 PEMS08"
  "PEMS04 PEMS07"
  "PEMS04 PEMS08"
  "PEMS07 PEMS08"
)

datasets=(PEMS03 PEMS04 PEMS07 PEMS08)

IFS=',' read -r -a gpu_ids <<< "$GPU_LIST"

if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
  echo "[ERROR] No GPUs provided. Use arg1 or GPU_LIST/GPU env var."
  exit 1
fi

if ! [[ "$MAX_JOBS_PER_GPU" =~ ^[0-9]+$ ]] || [[ "$MAX_JOBS_PER_GPU" -lt 1 ]]; then
  echo "[ERROR] MAX_JOBS_PER_GPU must be a positive integer."
  exit 1
fi

run_train() {
  local cfg="$1"
  local gpu="$2"
  echo "[RUN][GPU ${gpu}] python experiments/train.py -c ${cfg} -g ${gpu}"
  python experiments/train.py -c "${cfg}" -g "${gpu}"
}

run_pair_pipeline() {
  local pair="$1"
  local gpu="$2"

  local src_a src_b source_tag pre_cfg ckpt_root best_ckpt
  src_a=$(echo "$pair" | awk '{print $1}')
  src_b=$(echo "$pair" | awk '{print $2}')
  source_tag="${src_a}_${src_b}"

  echo "[PAIR][GPU ${gpu}] Start ${source_tag}"

  pre_cfg="baselines/NodeSTID/PEMS_Combined_v2_${source_tag}_E30.py"
  if [[ ! -f "$pre_cfg" ]]; then
    echo "[SKIP][GPU ${gpu}] Missing pretrain config: ${pre_cfg}"
    return 0
  fi

  run_train "$pre_cfg" "$gpu"

  if [[ "$PRETRAIN_ONLY" == "1" ]]; then
    echo "[PAIR][GPU ${gpu}] Done pretrain-only ${source_tag}"
    return 0
  fi

  ckpt_root="checkpoints/NodeSTIDv2/PEMS_Combined_v2_${source_tag}_10_12_12"
  if [[ ! -d "$ckpt_root" ]]; then
    echo "[SKIP][GPU ${gpu}] Pretrain checkpoint root not found: ${ckpt_root}"
    return 0
  fi

  best_ckpt=$(find "$ckpt_root" -type f -name "NodeSTIDv2_best_val_MAE.pt" | sort | head -n 1)
  if [[ -z "$best_ckpt" ]]; then
    echo "[SKIP][GPU ${gpu}] No best checkpoint found under: ${ckpt_root}"
    return 0
  fi

  echo "[INFO][GPU ${gpu}] Using pretrained checkpoint: ${best_ckpt}"

  for target in "${datasets[@]}"; do
    local gc_cfg
    if [[ "$target" == "$src_a" || "$target" == "$src_b" ]]; then
      continue
    fi

    gc_cfg="baselines/NodeSTID/${target}_GraphConditioned_From_${source_tag}_E30.py"
    if [[ ! -f "$gc_cfg" ]]; then
      echo "[SKIP][GPU ${gpu}] Missing graph config: ${gc_cfg}"
      continue
    fi

    echo "[RUN][GPU ${gpu}] Transfer ${source_tag} -> ${target}"
    NODESTID_PRETRAINED_PATH="$best_ckpt" run_train "$gc_cfg" "$gpu"
  done

  echo "[PAIR][GPU ${gpu}] Completed ${source_tag}"
}

available_slots=()
for gpu in "${gpu_ids[@]}"; do
  for ((i = 0; i < MAX_JOBS_PER_GPU; i++)); do
    available_slots+=("$gpu")
  done
done

if [[ "${#available_slots[@]}" -eq 0 ]]; then
  echo "[ERROR] No execution slots available."
  exit 1
fi

echo "[INFO] GPU list: ${GPU_LIST}"
echo "[INFO] MAX_JOBS_PER_GPU: ${MAX_JOBS_PER_GPU}"
echo "[INFO] Total parallel slots: ${#available_slots[@]}"

declare -A pid_to_gpu
declare -A pid_to_pair

wait_for_one() {
  local done_pid done_status done_gpu done_pair
  done_status=0
  if wait -n -p done_pid; then
    done_status=0
  else
    done_status=$?
  fi

  done_gpu="${pid_to_gpu[$done_pid]}"
  done_pair="${pid_to_pair[$done_pid]}"
  available_slots+=("$done_gpu")
  unset pid_to_gpu["$done_pid"]
  unset pid_to_pair["$done_pid"]

  if [[ "$done_status" -ne 0 ]]; then
    echo "[FAIL][GPU ${done_gpu}] ${done_pair} exited with code ${done_status}"
    return "$done_status"
  fi

  echo "[OK][GPU ${done_gpu}] ${done_pair}"
  return 0
}

failed=0

for pair in "${pairs[@]}"; do
  while [[ "${#available_slots[@]}" -eq 0 ]]; do
    if ! wait_for_one; then
      failed=1
    fi
  done

  gpu="${available_slots[0]}"
  available_slots=("${available_slots[@]:1}")

  (
    run_pair_pipeline "$pair" "$gpu"
  ) &
  pid=$!
  pid_to_gpu["$pid"]="$gpu"
  pid_to_pair["$pid"]="$pair"
  echo "[LAUNCH][GPU ${gpu}] ${pair} (pid=${pid})"
done

while [[ "${#pid_to_gpu[@]}" -gt 0 ]]; do
  if ! wait_for_one; then
    failed=1
  fi
done

if [[ "$failed" -ne 0 ]]; then
  echo "[DONE] Completed with failures."
  exit 1
fi

echo "[DONE] Cross-dataset matrix finished successfully."
