#!/usr/bin/env bash
# Run one step of fitting the cost models of the physical optimizers: the
# module MODULE of scripts/fitting with ARGS, on the chunk of its array task
# when submitted as an array. The step installs the environment, and Julia
# and its packages if it runs Julia, on its node's own disk, in a copy of the
# repository's tracked files there, so steps share nothing they could race to
# install, and read nothing of the environment from the shared filesystem,
# where starting Python alone takes minutes. The resources of a step come from
# the flags of sbatch, which scripts/fitting/launch_physical_data.sh gives
# each step. Submit from the repository, with absolute paths in ARGS.
#
#     sbatch [resources] scripts/fitting/step.slurm MODULE [ARGS...]
#SBATCH --job-name=physical-data
#SBATCH --nodes=1
# The environment, Julia's depot and the copy of the repository need the disk.
#SBATCH --tmp=20G
#SBATCH --output=physical-data-%j.log
#SBATCH --mail-type=FAIL

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

module=$1
shift
args=()
# Each submission of the generator draws new programs, unless ARGS give a seed.
if [[ $module == generate_*_data ]]; then
  args+=(--seed "${SLURM_ARRAY_JOB_ID:-0}")
fi
args+=("$@")
if [[ -n ${SLURM_ARRAY_TASK_ID:-} ]]; then
  args+=(--chunk "$SLURM_ARRAY_TASK_ID")
fi

local=$(mktemp -d "${TMPDIR:-/tmp}/physical-data-XXXXXX")
trap 'rm -rf "$local"' EXIT
echo "installing the environment in $local"
mkdir "$local/repo"
# git lists tracked files deleted from the checkout too, which tar can't copy.
git ls-files -z | while IFS= read -r -d '' file; do
  if [[ -e $file || -L $file ]]; then printf '%s\0' "$file"; fi
done | tar --null -T - -cf - |
  tar -xf - -C "$local/repo"
cd "$local/repo"
# pixi's package cache goes there too, rather than filling the home quota.
export PIXI_CACHE_DIR="$local/pixi-cache"
pixi install --frozen -e fitting
# Julia's depot, with its packages and their caches, compiled for this CPU,
# goes there too. The trailing colon keeps the depots of Julia's own
# installation, and leaves out the shared one.
export JULIA_DEPOT_PATH="$local/depot:"
if [[ $module == time_physical_data ]]; then
  echo "installing Julia and its packages in $local/depot"
  pixi run --frozen -e fitting python -c '
import logging, juliapkg
# juliapkg says what it installs, and precompiles, only at this level.
logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
juliapkg.resolve()
'
fi
pixi run --frozen -e fitting python -m "scripts.fitting.$module" "${args[@]}"
