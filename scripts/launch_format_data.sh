#!/usr/bin/env bash
# Submit every step of fitting the formatter's cost model to DIR. Each chunk is
# timed and featurized as soon as it is generated, and the model is fitted to
# all of them once they are done, and written to DIR/model.json. The programs
# and their timings and features go to DIR/data, logs to DIR/logs, and the IDs
# of the jobs to DIR/jobs.txt, which scripts/status_format_data.sh reads to
# show their progress. A failed step cancels the steps which depend on it.
# Every job is charged to the Slurm account A if given, or to the default
# account otherwise. With --resume, DIR's programs are kept rather than
# generated, only those not yet timed or featurized are, and the jobs are added
# to DIR/jobs.txt. Options of this script come before DIR, and those after it
# are generate_format_data's.
#
#     scripts/launch_format_data.sh [--account A] [--chunks N] [--repeat N] \
#         [--resume] DIR [generate_format_data options]
set -euo pipefail

usage="usage: $0 [--account A] [--chunks N] [--repeat N] [--resume] DIR \
[generate_format_data options]"
account=
chunks=8
repeat=5
resume=
while [[ $# -gt 0 ]]; do
  case $1 in
    --account) account=$2; shift 2 ;;
    --account=*) account=${1#*=}; shift ;;
    --resume) resume=--resume; shift ;;
    --chunks) chunks=$2; shift 2 ;;
    --repeat) repeat=$2; shift 2 ;;
    -h | --help) echo "$usage"; exit 0 ;;
    *) break ;;
  esac
done
if [[ $# -lt 1 ]]; then
  echo "$usage" >&2
  exit 2
fi
mkdir -p "$1/logs"
directory=$(cd -- "$1" && pwd)
shift
if [[ $resume && $# -gt 0 ]]; then
  echo "$0: --resume generates nothing, so takes no generate options: $*" >&2
  exit 2
fi
# Generating again would replace programs whose timings and features remain.
if [[ ! $resume ]] && compgen -G "$directory/data/*.program.json" >/dev/null; then
  echo "$0: $directory already has programs; pass --resume before it" >&2
  exit 2
fi

cd -- "$(dirname -- "$0")"
repo_directory=$(git rev-parse --show-toplevel)
cd -- "$repo_directory"

# Install the environment and precompile Julia's packages here, once, rather
# than in every task at once, which race to precompile them into the shared
# depot. Julia caches precompiled packages for the CPU which compiles them, so
# compile them for the targets of Julia's own x86 builds, which every node can
# load. This doesn't change the code Julia compiles while running, such as the
# kernels being timed, which is always for the CPU it runs on. juliapkg only
# records that it installed Julia's packages, so install any which the depot
# lacks, such as a depot of another machine, at the versions it recorded.
export JULIA_CPU_TARGET=${JULIA_CPU_TARGET:-"generic;sandybridge,-xsaveopt,clone_all;haswell,-rdrnd,base(1);x86-64-v4,-rdrnd,base(1)"}
pixi install --frozen -e fitting
pixi run --frozen -e fitting python -c '
import subprocess, juliapkg
subprocess.run(
    [juliapkg.executable(), f"--project={juliapkg.project()}", "-e",
     "using Pkg; Pkg.instantiate()"],
    check=True,
)
from finch.compile_jl.julia import init_julia
init_julia()
'

submit() {
  local id
  id=$(sbatch --parsable --chdir="$repo_directory" --kill-on-invalid-dep=yes \
    ${account:+"--account=$account"} "$@")
  echo "${id%%;*}"
}

jobs=$directory/jobs.txt
array=--array=0-$((chunks - 1))
logs=$directory/logs
generate=
if [[ ! $resume ]]; then
  : >"$jobs"
  generate=$(submit "$array" --output="$logs/generate-%A_%a.log" \
    scripts/generate_format_data.slurm "$directory" "$@")
  echo "generate $generate" >>"$jobs"
fi
time=$(submit "$array" ${generate:+"--dependency=aftercorr:$generate"} \
  --output="$logs/time-%A_%a.log" \
  scripts/time_format_data.slurm "$directory" --repeat "$repeat" $resume)
echo "time $time" >>"$jobs"
featurize=$(submit "$array" ${generate:+"--dependency=aftercorr:$generate"} \
  --output="$logs/featurize-%A_%a.log" \
  scripts/featurize_format_data.slurm "$directory" $resume)
echo "featurize $featurize" >>"$jobs"
fit=$(submit --dependency="afterok:$time:$featurize" \
  --output="$logs/fit-%j.log" \
  scripts/fit_format_data.slurm "$directory" "$directory/model.json")
echo "fit $fit" >>"$jobs"

echo "submitted $chunks chunks:"
sed 's/^/  /' "$jobs"
echo "check progress with: scripts/status_format_data.sh $directory"
