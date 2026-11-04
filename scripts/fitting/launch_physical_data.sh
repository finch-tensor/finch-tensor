#!/usr/bin/env bash
# Submit every step of fitting the cost model of the stage STAGE of
# scripts/fitting/physical_bases.py, by default protocol, to DIR, with the
# models of the stages below it given as --model NAME=PATH, or the built-in
# ones. Each chunk is timed and featurized with the basis of the stage, and
# with every basis of --basis, as soon as it is generated, and once they are
# done the features of each basis and the timings are condensed into
# DIR/NAME.json, and a model is fitted to them and written to
# DIR/NAME.model.json, which the next stage takes as its --model. The programs
# and their timings and features go to DIR/data, logs to DIR/logs, and the IDs
# of the jobs to DIR/jobs.txt, which scripts/fitting/status_physical_data.sh
# reads to show their progress. A failed step cancels the steps which depend
# on it. Every job is charged to the Slurm account A if given, or to the
# default account otherwise. With --resume, DIR's programs are kept rather
# than generated, only the variants not yet timed are timed, each example is
# featurized only with the bases it lacks, and the jobs are added to
# DIR/jobs.txt. Each chunk is timed and featurized for --hours H, by default 6,
# and stops then, since its programs cost too differently to choose how many
# fill the time, so more are generated than any chunk could time. Options of
# this script come before DIR, and those after it are generate_physical_data's.
#
#     scripts/fitting/launch_physical_data.sh [--account A] [--stage STAGE] \
#         [--model NAME=PATH]... [--basis NAME]... [--chunks N] [--repeat N] \
#         [--hours H] [--resume] DIR [generate_physical_data options]
set -euo pipefail

usage="usage: $0 [--account A] [--stage STAGE] [--model NAME=PATH]... \
[--basis NAME]... [--chunks N] [--repeat N] [--hours H] [--resume] DIR \
[generate_physical_data options]"
account=
stage=protocol
models=()
bases=()
chunks=8
repeat=5
hours=6
resume=
while [[ $# -gt 0 ]]; do
  case $1 in
    --account) account=$2; shift 2 ;;
    --account=*) account=${1#*=}; shift ;;
    --stage) stage=$2; shift 2 ;;
    --stage=*) stage=${1#*=}; shift ;;
    --model) models+=(--model "$2"); shift 2 ;;
    --model=*) models+=(--model "${1#*=}"); shift ;;
    --basis) bases+=("$2"); shift 2 ;;
    --basis=*) bases+=("${1#*=}"); shift ;;
    --resume) resume=--resume; shift ;;
    --chunks) chunks=$2; shift 2 ;;
    --repeat) repeat=$2; shift 2 ;;
    --hours) hours=$2; shift 2 ;;
    -h | --help) echo "$usage"; exit 0 ;;
    *) break ;;
  esac
done
if [[ $# -lt 1 ]]; then
  echo "$usage" >&2
  exit 2
fi
# The basis of the stage is always fitted, and each basis is fitted once.
unique=()
basis_options=()
for basis in "$stage" "${bases[@]}"; do
  if [[ " ${unique[*]} " != *" $basis "* ]]; then
    unique+=("$basis")
    basis_options+=(--basis "$basis")
  fi
done
bases=("${unique[@]}")
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

# Install the environment and Julia's packages here, once, rather than in
# every task at once, which would race to install them into the shared depot.
# Nothing is precompiled here: each task which runs Julia precompiles its
# packages for its own CPU into a depot on its node's own disk, see
# scripts/fitting/step.slurm, so the pipeline neither needs nor rewrites the
# caches of the shared depot. juliapkg only records that it installed Julia's
# packages, so install any which the depot lacks, such as a depot of another
# machine, at the versions it recorded, without precompiling them.
pixi install --frozen -e fitting
echo "installing Julia's packages"
JULIA_PKG_PRECOMPILE_AUTO=0 pixi run --frozen -e fitting python -c '
import subprocess, juliapkg
subprocess.run(
    [juliapkg.executable(), f"--project={juliapkg.project()}", "-e",
     "using Pkg; Pkg.instantiate()"],
    check=True,
)
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
step=scripts/fitting/step.slurm
if [[ ! $resume ]]; then
  : >"$jobs"
  generate=$(submit "$array" --job-name=generate --cpus-per-task=8 --mem=64G \
    --time=02:00:00 --output="$logs/generate-%A_%a.log" \
    "$step" generate_physical_data "$directory" --stage "$stage" "$@")
  echo "generate $generate" >>"$jobs"
fi
# Each timing task takes a whole node, so other jobs don't disturb it. A task
# stops starting variants after its hours, and the limit leaves two more for
# starting Julia and finishing the last variant, which can run for minutes.
limit=$((hours + 2)):00:00
time=$(submit "$array" ${generate:+"--dependency=aftercorr:$generate"} \
  --job-name=time --exclusive --time="$limit" --output="$logs/time-%A_%a.log" \
  "$step" time_physical_data "$directory" "${models[@]}" --repeat "$repeat" \
  --hours "$hours" $resume)
echo "time $time" >>"$jobs"
featurize=$(submit "$array" ${generate:+"--dependency=aftercorr:$generate"} \
  --job-name=featurize --cpus-per-task=2 --mem=16G --time="$limit" \
  --output="$logs/featurize-%A_%a.log" \
  "$step" featurize_physical_data "$directory" "${models[@]}" \
  "${basis_options[@]}" --hours "$hours" $resume)
echo "featurize $featurize" >>"$jobs"
for basis in "${bases[@]}"; do
  condense=$(submit --dependency="afterok:$time:$featurize" \
    --job-name="condense-$basis" --cpus-per-task=1 --mem=16G --time=01:00:00 \
    --output="$logs/condense-$basis-%j.log" \
    "$step" condense_physical_data "$directory" --basis "$basis")
  echo "condense-$basis $condense" >>"$jobs"
  fit=$(submit --dependency="afterok:$condense" --job-name="fit-$basis" \
    --cpus-per-task=1 --mem=16G --time=01:00:00 --mail-type=END,FAIL \
    --output="$logs/fit-$basis-%j.log" \
    "$step" fit_physical_data "$directory/$basis.json" "$directory/$basis.model.json")
  echo "fit-$basis $fit" >>"$jobs"
done

echo "submitted $chunks chunks of the $stage stage:"
sed 's/^/  /' "$jobs"
echo "check progress with: scripts/fitting/status_physical_data.sh $directory"
