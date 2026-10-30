#!/usr/bin/env bash
# Generate programs for fitting the formatter's cost model, one chunk per array
# task, into a directory which all the chunks share:
#
#     sbatch scripts/competition-cpu.slurm DIR [--examples N] [--lg-max-space N] \
#         [--lg-max-dim N] [--seed S]
#
# The seed defaults to the job ID, so each submission draws new programs.
#SBATCH --job-name=finch-generate-format-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --array=0-63
#SBATCH --output=generate-format-data-%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahrens@gatech.edu

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: sbatch $0 DIR [generate_format_data options]" >&2
  exit 2
fi

# Slurm runs a spooled copy of this script, so find the repository from the
# submission directory rather than from $0.
repo_directory="${FINCH_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi
repo_directory=$(git -C "$repo_directory" rev-parse --show-toplevel)

# Options after the seed override it, and the chunk is always the task's.
cd "$repo_directory"
pixi run python -m finch.autoschedule.fitting.generate_format_data \
  --seed "${SLURM_ARRAY_JOB_ID:-0}" "$@" --chunk "${SLURM_ARRAY_TASK_ID:-0}"
