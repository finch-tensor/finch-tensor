#!/usr/bin/env bash
# Extract the features of one or more bases of the programs of one chunk of DIR
# per array task.
# Submit from the repository, or run every step with
# scripts/launch_protocol_data.sh.
#
#     sbatch scripts/featurize_protocol_data.slurm DIR [--basis NAME]... [--resume]
#SBATCH --job-name=featurize-protocol-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=300G
#SBATCH --time=16:00:00
#SBATCH --array=0-7
#SBATCH --output=featurize-protocol-data-%A_%a.log
#SBATCH --mail-type=FAIL

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pixi run --frozen -e fitting python -m scripts.featurize_protocol_data \
  "$@" --chunk "${SLURM_ARRAY_TASK_ID:-0}"
