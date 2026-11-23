#!/usr/bin/env bash
set -euo pipefail

wmj_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
wmj_repo_root="$(cd -- "${wmj_script_dir}/../../.." && pwd)"
wmj_python="${WMJ_PYTHON:-/home/yangzhe/miniconda3/envs/pingpong/bin/python}"
wmj_gpu_uuid="${WMJ_GPU_UUID:-GPU-e74458ec-002a-1bac-e0be-d0a5713b661e}"
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <new-output-directory>" >&2
  exit 2
fi
wmj_output="$1"
if [[ "${wmj_output}" != /* ]]; then wmj_output="${wmj_repo_root}/${wmj_output}"; fi
if [[ -e "${wmj_output}" || -e "${wmj_output}_scope" ]]; then
  echo "Refusing to overwrite tangential-prior diagnostic or resource logs" >&2
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
  "${wmj_python}" -m world_model.audit_tangent_replay \
  --output-dir "${wmj_output}" --gpu-uuid "${wmj_gpu_uuid}"
