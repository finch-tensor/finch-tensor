#!/usr/bin/env bash
# Run one step of fitting the cost models of the physical optimizers: the
# module MODULE of scripts/fitting with ARGS, on the chunk of its array task
# when submitted as an array. The resources of a step come from the flags of
# sbatch, which scripts/fitting/launch_physical_data.sh gives each step.
# Submit from the repository.
#
#     sbatch [resources] scripts/fitting/step.slurm MODULE [ARGS...]
#SBATCH --job-name=physical-data
#SBATCH --nodes=1
#SBATCH --output=physical-data-%j.log
#SBATCH --mail-type=FAIL

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

module=$1
shift
args=()
# Each submission of the generator draws new programs, unless ARGS give a seed.
if [[ $module == generate_physical_data ]]; then
  args+=(--seed "${SLURM_ARRAY_JOB_ID:-0}")
fi
args+=("$@")
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  args+=(--chunk "$SLURM_ARRAY_TASK_ID")
fi
pixi run --frozen -e fitting python -m "scripts.fitting.$module" "${args[@]}"
