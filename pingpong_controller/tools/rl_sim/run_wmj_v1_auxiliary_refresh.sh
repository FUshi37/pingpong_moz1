#!/usr/bin/env bash
set -euo pipefail

wmj_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
wmj_repo_root="$(cd -- "${wmj_script_dir}/../../.." && pwd)"
wmj_python="${WMJ_PYTHON:-/home/yangzhe/miniconda3/envs/pingpong/bin/python}"
wmj_gpu_uuid="${WMJ_GPU_UUID:-GPU-e74458ec-002a-1bac-e0be-d0a5713b661e}"
if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <collect|fit> <input-directory> <new-output-directory> [worker options]" >&2
  exit 2
fi
wmj_mode="$1"
wmj_input="$2"
wmj_output="$3"
shift 3
case "${wmj_mode}" in
  collect) wmj_module="collect_refresh_labels"; wmj_input_flag="--collection"; wmj_wall=1800 ;;
  fit) wmj_module="fit_auxiliary_refresh"; wmj_input_flag="--labels"; wmj_wall=900 ;;
  *) echo "Unknown diagnostic mode" >&2; exit 2 ;;
esac
if [[ "${wmj_input}" != /* ]]; then wmj_input="${wmj_repo_root}/${wmj_input}"; fi
if [[ "${wmj_output}" != /* ]]; then wmj_output="${wmj_repo_root}/${wmj_output}"; fi
if [[ -e "${wmj_output}" || -e "${wmj_output}_scope" ]]; then
  echo "Refusing to overwrite diagnostic or resource logs" >&2
  exit 2
fi
cd -- "${wmj_script_dir}"
exec "${wmj_python}" training_memory_scope.py \
  --wmj-v1-conditional-psi \
  --log-dir "${wmj_output}_scope" --memory-high-gib 10 --memory-max-gib 11 \
  --memory-stop-gib 9.7 --host-stop-available-gib 6 --host-stop-streak-samples 2 \
  -- bash "${wmj_script_dir}/run_wmj_v1_collect.sh" --scope-child "${wmj_output}_scope/console.log" \
  env PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="${wmj_gpu_uuid}" \
  JAX_PLATFORMS=cuda XLA_PYTHON_CLIENT_PREALLOCATE=false OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  "${wmj_python}" -m "world_model.${wmj_module}" "${wmj_input_flag}" "${wmj_input}" \
  --output-dir "${wmj_output}" --gpu-uuid "${wmj_gpu_uuid}" --max-wall-s "${wmj_wall}" "$@"
