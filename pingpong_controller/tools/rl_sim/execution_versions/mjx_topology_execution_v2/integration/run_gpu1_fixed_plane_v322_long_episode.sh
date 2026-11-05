#!/usr/bin/env bash
set -euo pipefail
[[ $# -ge 2 && $# -le 3 ]] || { echo 'usage: launcher --formal|--smoke|--reward-probe UNIQUE_RUN_ID [V322_RESUME_CHECKPOINT]'; exit 2; }
V322_KIND=$1
V322_RUN_ID=$2
[[ $V322_KIND == --formal || $V322_KIND == --smoke || $V322_KIND == --reward-probe ]] || exit 2
[[ $V322_RUN_ID =~ ^[a-zA-Z0-9_-]+$ ]] || exit 2
V322_SOLVER_SCHEDULE=${V322_MJX_LINE_SEARCH_SCHEDULE:-legacy}
V322_EXECUTION_MODE=${V322_MJX_EXECUTION_MODE:-legacy}
[[ $V322_EXECUTION_MODE == legacy || $V322_EXECUTION_MODE == topology_v2 ]] || {
  echo 'Unknown MJX execution mode'; exit 2;
}
[[ $V322_EXECUTION_MODE == legacy || $V322_SOLVER_SCHEDULE == legacy ]] || {
  echo 'Choose one MJX execution option; do not stack schedules'; exit 2;
}
[[ $V322_SOLVER_SCHEDULE == legacy || $V322_SOLVER_SCHEDULE == chunk5_v1 ]] || {
  echo 'Unknown MJX line-search schedule'; exit 2;
}
[[ $V322_KIND != --reward-probe || ( $V322_SOLVER_SCHEDULE == legacy && $V322_EXECUTION_MODE == legacy ) ]] || {
  echo 'MJX scheduling opt-in applies to training/smoke only'; exit 2;
}
V322_SIM=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
V322_REPO=$(cd -- "$V322_SIM/../../.." && pwd)
V322_PYTHON=/home/yangzhe/miniconda3/envs/pingpong/bin/python
V322_GPU=GPU-e74458ec-002a-1bac-e0be-d0a5713b661e
V322_RUNTIME=/dev/shm/pp_v322_$V322_RUN_ID
V322_MODELS=$V322_REPO/pingpong_controller/outputs/rl_sim/$V322_RUN_ID
[[ ! -e $V322_RUNTIME && ! -e $V322_MODELS ]] || { echo 'Run identity already exists'; exit 2; }
if [[ $V322_KIND == --formal ]]; then
  [[ -n ${TMUX:-} && $(tmux display-message -p -t "$TMUX_PANE" '#S') == pp_gpu1 ]] || {
    echo 'Formal training belongs in the existing pp_gpu1 pane'; exit 2;
  }
fi
V322_PIDS=$(nvidia-smi -i "$V322_GPU" --query-compute-apps=pid --format=csv,noheader)
[[ -z $V322_PIDS ]] || { echo "GPU1 occupied: $V322_PIDS"; exit 2; }
V322_AVAILABLE_KIB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
V322_TEMP=$(nvidia-smi -i "$V322_GPU" --query-gpu=temperature.gpu --format=csv,noheader,nounits)
(( V322_AVAILABLE_KIB >= 13824*1024 && V322_TEMP < 78 )) || {
  echo 'Insufficient measured combined host budget or GPU too hot'; exit 2;
}
mkdir -m 700 "$V322_RUNTIME"
mkdir "$V322_RUNTIME/source" "$V322_RUNTIME/tmp" "$V322_RUNTIME/wandb" \
  "$V322_RUNTIME/wandb_cache" "$V322_RUNTIME/wandb_config" "$V322_RUNTIME/wandb_data"
for V322_FILE in train_fixed_plane_v322.py fixed_plane_v322_long_episode.py fixed_plane_v321_long_episode.py \
  fixed_plane_v322_condition_gates.py fixed_plane_v322_late_cycle_reward.py fixed_plane_v322_quality_repair.py \
  train_juggle_mjx_ppo.py bounded_validation_memory.py training_memory_scope.py packed_rollout_metrics.py \
  mjx_line_search_schedule.py mjx_topology_execution.py; do
  cp -- "$V322_SIM/$V322_FILE" "$V322_RUNTIME/source/$V322_FILE"
done
cp -- "${BASH_SOURCE[0]}" "$V322_RUNTIME/source/launcher.sh"
git -C "$V322_REPO" rev-parse HEAD > "$V322_RUNTIME/source/git_head.txt"
git -C "$V322_REPO" status --short --branch --untracked-files=no > "$V322_RUNTIME/source/git_status.txt"
V322_COMMAND=("$V322_PYTHON" -B -u "$V322_RUNTIME/source/train_fixed_plane_v322.py"
  --repo-root "$V322_REPO" --runtime "$V322_RUNTIME" --save-dir "$V322_MODELS" --run-id "$V322_RUN_ID")
if [[ $V322_KIND == --smoke ]]; then V322_COMMAND+=(--smoke); fi
if [[ $V322_KIND == --reward-probe ]]; then V322_COMMAND+=(--reward-probe); fi
if [[ $# -eq 3 ]]; then V322_COMMAND+=(--resume "$3"); fi
if [[ -n ${V322_REWARD_PRESET:-} ]]; then
  V322_COMMAND+=(--reward-preset "$V322_REWARD_PRESET")
fi
if [[ ${V322_ALLOW_REWARD_CHANGE:-NO} == YES ]]; then
  V322_COMMAND+=(--allow-reward-change)
fi
if [[ -n ${V322_LOSS_PRESET:-} ]]; then
  V322_COMMAND+=(--loss-preset "$V322_LOSS_PRESET")
fi
if [[ ${V322_ALLOW_LOSS_CHANGE:-NO} == YES ]]; then
  V322_COMMAND+=(--allow-loss-change)
fi
if [[ -n ${V322_GATE_PRESET:-} ]]; then
  V322_COMMAND+=(--gate-preset "$V322_GATE_PRESET")
fi
if [[ ${V322_ALLOW_GATE_CHANGE:-NO} == YES ]]; then
  V322_COMMAND+=(--allow-gate-change)
fi
if [[ ${V322_PACKED_HOST_METRICS:-NO} == YES ]]; then
  V322_COMMAND+=(--packed-host-metrics)
fi
if [[ $V322_SOLVER_SCHEDULE != legacy ]]; then
  V322_COMMAND+=(--mjx-line-search-schedule "$V322_SOLVER_SCHEDULE")
fi
if [[ $V322_EXECUTION_MODE != legacy ]]; then
  V322_COMMAND+=(--mjx-execution-mode "$V322_EXECUTION_MODE")
fi
if [[ ${V322_QUALITY_REPAIR:-NO} == YES ]]; then
  [[ -n ${V322_LONG_QUALIFICATION_AUDIT:-} && -f $V322_LONG_QUALIFICATION_AUDIT ]]
  V322_COMMAND+=(--quality-repair --long-qualification-audit "$V322_LONG_QUALIFICATION_AUDIT")
fi
printf '%q ' "${V322_COMMAND[@]}" > "$V322_RUNTIME/command.sh"
printf '\n' >> "$V322_RUNTIME/command.sh"
# A systemd user service does not inherit the invoking shell's proxy. Pass
# only its already-configured networking variables to this V322 child;
# do not change the shared memory guard or any other GPU's environment.
V322_NETWORK_ENV=()
for V322_PROXY_NAME in http_proxy https_proxy all_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY no_proxy NO_PROXY; do
  if [[ -n ${!V322_PROXY_NAME:-} ]]; then
    V322_NETWORK_ENV+=("$V322_PROXY_NAME=${!V322_PROXY_NAME}")
  fi
done
cd "$V322_RUNTIME/tmp"
# The new GPU1 guard triggers before the existing GPU0 5-GiB host guard.
# Kernel limits remain tighter than the historical 11/12-GiB trainer boundary.
"$V322_PYTHON" -B "$V322_RUNTIME/source/training_memory_scope.py" \
  --log-dir "$V322_RUNTIME/guard" --memory-high-gib 10 --memory-max-gib 11 \
  --memory-stop-gib 9.7 --host-stop-available-gib 6 --host-stop-streak-samples 2 \
  -- env "${V322_NETWORK_ENV[@]}" CUDA_DEVICE_ORDER=PCI_BUS_ID "CUDA_VISIBLE_DEVICES=$V322_GPU" \
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_ENABLE_COMPILATION_CACHE=false \
  XLA_FLAGS='--xla_gpu_deterministic_ops=true --xla_gpu_force_compilation_parallelism=1 --xla_gpu_enable_llvm_module_compilation_parallelism=false' \
  OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2 PYTHONDONTWRITEBYTECODE=1 \
  "TMPDIR=$V322_RUNTIME/tmp" "MPLCONFIGDIR=$V322_RUNTIME/matplotlib" \
  WANDB_MODE=online WANDB_DISABLE_CODE=true WANDB_CONSOLE=off WANDB_INIT_TIMEOUT=180 \
  "WANDB_DIR=$V322_RUNTIME/wandb" "WANDB_CACHE_DIR=$V322_RUNTIME/wandb_cache" \
  "WANDB_CONFIG_DIR=$V322_RUNTIME/wandb_config" "WANDB_DATA_DIR=$V322_RUNTIME/wandb_data" \
  "${V322_COMMAND[@]}" 2>&1 | tee "$V322_RUNTIME/stdout_stderr.log"
