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
# Julia maps the caches of its precompiled packages into memory from its
# depot, and any process which precompiles them again into the same depot can
# delete a cache that a running step still maps, killing it with a bus error.
# So the step which runs Julia precompiles its packages, for the CPU of its
# node, into a depot on the node's own disk, which comes first in the depot
# path, and finds the packages in the shared depot behind it. It starts from a
# copy of the shared depot's caches, in case any are loadable here, so none
# are mapped from the shared depot. The trailing colon keeps the depots of
# Julia's own installation.
if [[ $module == time_physical_data ]]; then
  shared_depot=${JULIA_DEPOT_PATH:-$HOME/.julia}
  local_depot=$(mktemp -d "${TMPDIR:-/tmp}/julia-depot-XXXXXX")
  trap 'rm -rf "$local_depot"' EXIT
  read -r julia project < <(pixi run --frozen -e fitting python -c \
    'import juliapkg; print(juliapkg.executable(), juliapkg.project())' | tail -1)
  version=$("$julia" --startup-file=no -e 'print(VERSION.major, ".", VERSION.minor)')
  compiled=${shared_depot%%:*}/compiled/v$version
  mkdir -p "$local_depot/compiled" "$local_depot/registries"
  if [[ -d $compiled ]]; then
    cp -a "$compiled" "$local_depot/compiled/"
  fi
  export JULIA_DEPOT_PATH="$local_depot:$shared_depot:"
  echo "precompiling Julia's packages into $local_depot"
  "$julia" --startup-file=no "--project=$project" -e 'using Pkg; Pkg.precompile()'
fi
pixi run --frozen -e fitting python -m "scripts.fitting.$module" "${args[@]}"
