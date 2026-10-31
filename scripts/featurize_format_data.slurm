#!/usr/bin/env bash
# Extract the features of the programs of one chunk of DIR per array task.
# Submit from the repository, or run every step with
# scripts/launch_format_data.sh.
#
#     sbatch scripts/featurize_format_data.slurm DIR [--resume]
#SBATCH --job-name=featurize-format-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=featurize-format-data-%A_%a.log
#SBATCH --mail-type=FAIL

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pixi run --frozen -e fitting python -m scripts.featurize_format_data \
  "$@" --chunk "${SLURM_ARRAY_TASK_ID:-0}"
