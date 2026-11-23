#!/usr/bin/env bash
set -euo pipefail
[[ $# -eq 1 && ( $1 == --probe || $1 == --formal ) ]] || exit 2
KIND=${1#--}
ROOT=/home/yangzhe/Project/pingpong_controller
SIM="$ROOT/pingpong_controller/tools/rl_sim"
OUTPUT="$ROOT/pingpong_controller/outputs/rl_sim"
DESIGN="$OUTPUT/v226_stability_recovery_design_20261018_r1"
PYTHON_BIN=/home/yangzhe/miniconda3/envs/pingpong/bin/python
GPU_UUID=GPU-e74458ec-002a-1bac-e0be-d0a5713b661e
V226_SEED=${V226_SEED:-20266810}
RUN_ID=${RUN_ID:-fplc_v226_gpu1_${KIND}_seed${V226_SEED}_$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)}
[[ $RUN_ID =~ ^[a-zA-Z0-9._-]+$ ]] || exit 2
RUN="$OUTPUT/$RUN_ID"
[[ ! -e $RUN && -n ${TMUX:-} ]] || exit 2
[[ $(tmux display-message -p -t "$TMUX_PANE" '#S') == pp_gpu1 ]] || exit 2
[[ $(nvidia-smi -i 1 --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]') == "$GPU_UUID" ]] || exit 2
[[ -z $(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader | tr -d '[:space:]') ]] || exit 2
GPU_TEMP=$(nvidia-smi -i 1 --query-gpu=temperature.gpu --format=csv,noheader,nounits | tr -d '[:space:]')
AVAILABLE_KIB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
DISK_KIB=$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')
(( GPU_TEMP < 78 && AVAILABLE_KIB >= 16*1024*1024 && DISK_KIB >= 5*1024*1024 )) || exit 2
ADMISSION=()
if [[ $KIND == formal ]]; then
 [[ -n ${V226_ADMISSION:-} && -f $V226_ADMISSION ]] || exit 2
 ADMISSION=(--admission "$V226_ADMISSION")
fi
env CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
 "$PYTHON_BIN" "$SIM/prepare_fixed_plane_v226.py" --run-dir "$RUN" --kind "$KIND" \
 --launcher "${BASH_SOURCE[0]}" --seed "$V226_SEED" "${ADMISSION[@]}"
nvidia-smi -i 1 --query-gpu=index,uuid,name,memory.total --format=csv,noheader > "$RUN/gpu_identity.txt"
FROZEN="$RUN/source_snapshot/pingpong_controller/tools/rl_sim"
cd "$FROZEN"
exec "$PYTHON_BIN" "$FROZEN/training_memory_scope.py" --log-dir "$RUN/memory_scope" \
 --memory-high-gib 11 --memory-max-gib 12 --memory-stop-gib 10.5 \
 --host-stop-available-gib 4 --host-stop-streak-samples 3 \
 -- env GUARD_SWAP_PRESSURE_AVAILABLE_KB=4194304 \
 "$FROZEN/run_with_host_memory_guard.sh" "$RUN/stdout_stderr.log" \
 env CUDA_DEVICE_ORDER=PCI_BUS_ID "CUDA_VISIBLE_DEVICES=$GPU_UUID" JAX_PLATFORMS=cuda PYTHONUNBUFFERED=1 \
 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS=--xla_gpu_deterministic_ops=true \
 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MALLOC_ARENA_MAX=2 "MPLCONFIGDIR=$RUN/matplotlib" \
 "JAX_COMPILATION_CACHE_DIR=$DESIGN/compilation_cache" WANDB_DISABLE_CODE=true WANDB_INIT_TIMEOUT=180 \
 "WANDB_DIR=$RUN/wandb" "WANDB_CACHE_DIR=$RUN/wandb_cache" "WANDB_CONFIG_DIR=$RUN/wandb_config" \
 bash "$RUN/command.sh"
