#!/usr/bin/env bash
set -euo pipefail
[[ ${V322_REWARD_PRESET:-late_center_precision} == late_center_precision &&
   ${V322_LOSS_PRESET:-anchor1} == anchor1 &&
   ${V322_GATE_PRESET:-deployment_groups} == deployment_groups ]] || {
  echo 'Execution v2 requires the validated V322 preset tuple'; exit 2;
}
export V322_REWARD_PRESET=late_center_precision
export V322_LOSS_PRESET=anchor1
export V322_GATE_PRESET=deployment_groups
export V322_MJX_EXECUTION_MODE=topology_v2
V322_EXECUTION_SIM=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "$V322_EXECUTION_SIM/run_gpu1_fixed_plane_v322_long_episode.sh" "$@"
