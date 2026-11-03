#!/usr/bin/env bash
# Time the programs of one chunk of DIR per array task. Submit from the
# repository, or run every step with scripts/launch_protocol_data.sh.
#
#     sbatch scripts/time_protocol_data.slurm DIR [--repeat N] [--resume]
#
# Each task takes a whole node, so other jobs don't disturb its timings.
#SBATCH --job-name=time-protocol-data
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=time-protocol-data-%A_%a.log
#SBATCH --mail-type=FAIL

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pixi run --frozen -e fitting python -m scripts.time_protocol_data \
  "$@" --chunk "${SLURM_ARRAY_TASK_ID:-0}"
