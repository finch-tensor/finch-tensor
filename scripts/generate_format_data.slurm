#!/usr/bin/env bash
# Generate the programs of one chunk of DIR per array task. Submit from the
# repository, or run every step with scripts/launch_format_data.sh.
#
#     sbatch scripts/generate_format_data.slurm DIR [--programs N] \
#         [--samples N] [--keep N] [--lg-max-space N] [--lg-max-dim N] [--seed S]
#
# The seed defaults to the job ID, so each submission draws new programs.
#SBATCH --job-name=generate-format-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --output=generate-format-data-%A_%a.log
#SBATCH --mail-type=FAIL

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Options after the seed override it, and the chunk is always the task's.
pixi run --frozen -e fitting python -m scripts.generate_format_data \
  --seed "${SLURM_ARRAY_JOB_ID:-0}" "$@" --chunk "${SLURM_ARRAY_TASK_ID:-0}"
