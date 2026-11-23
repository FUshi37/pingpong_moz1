#!/usr/bin/env bash
set -euo pipefail

wmj_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
wmj_repo_root="$(cd -- "${wmj_script_dir}/../../.." && pwd)"
wmj_python="${WMJ_PYTHON:-/home/yangzhe/miniconda3/envs/pingpong/bin/python}"
wmj_gpu_uuid="${WMJ_GPU_UUID:-GPU-e74458ec-002a-1bac-e0be-d0a5713b661e}"
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 <base|fit|evaluate> <new-output-directory> [worker options]" >&2
  exit 2
fi
wmj_mode="$1"
wmj_output="$2"
shift 2
case "${wmj_mode}" in base|fit|evaluate) ;; *) echo "Unknown mode" >&2; exit 2 ;; esac
if [[ "${wmj_output}" != /* ]]; then wmj_output="${wmj_repo_root}/${wmj_output}"; fi
if [[ -e "${wmj_output}" || -e "${wmj_output}_scope" ]]; then
  echo "Refusing to overwrite normal-prior experiment or resource logs" >&2
  exit 2
fi
wmj_evidence="${wmj_repo_root}/pingpong_controller/outputs/rl_sim/wmj_v1"
cd -- "${wmj_script_dir}"
exec "${wmj_python}" training_memory_scope.py \
  --wmj-v1-conditional-psi \
  --log-dir "${wmj_output}_scope" --memory-high-gib 10 --memory-max-gib 11 \
  --memory-stop-gib 9.7 --host-stop-available-gib 6 --host-stop-streak-samples 2 \
  -- bash "${wmj_script_dir}/run_wmj_v1_collect.sh" --scope-child "${wmj_output}_scope/console.log" \
  env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="${wmj_gpu_uuid}" \
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "${wmj_python}" -m world_model.normal_experiment --mode "${wmj_mode}" \
  --support-replay "${wmj_evidence}/support_replay_full_a01" \
  --robot-checkpoint "${wmj_evidence}/support_robot_fit_a02/robot_support_u006144.json" \
  --contact-checkpoint "${wmj_evidence}/contact_activation_control_a01/contact_coupled_u014336.json" \
  --auxiliary-checkpoint "${wmj_evidence}/solver_aux_fit_a01/auxiliary_u004096.json" \
  --output-dir "${wmj_output}" --gpu-uuid "${wmj_gpu_uuid}" --max-wall-s 1800 "$@"
