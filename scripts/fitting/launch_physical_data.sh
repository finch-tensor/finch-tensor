#!/usr/bin/env bash
# Submit every step of fitting the cost model of the stage STAGE of
# scripts/fitting/physical_bases.py, by default protocol, to DIR, with the
# models of the stages below it given as --model NAME=PATH, or the built-in
# ones. Each step installs its environment on its node's own disk, see
# scripts/fitting/step.slurm. Each chunk is timed as soon as it is generated,
# and once every chunk is done the timed programs and their timings are
# coalesced into DIR/examples.json, and a model of the basis of the stage, and
# of every basis of --basis, is fitted to them, featurizing them as they were
# timed, and written to DIR/NAME.model.json, which the next stage takes as its
# --model. The examples file is small, so scripts/fitting/fit_physical_data.py
# can fit it again anywhere, without DIR/data. The programs and their timings
# go to DIR/data, logs to DIR/logs, and the IDs of the jobs to DIR/jobs.txt,
# which scripts/fitting/status_physical_data.sh reads to show their progress.
# The coalescing waits for the timing to end however it ends, since the chunks
# which ran out of time still count, and otherwise a failed step cancels the
# steps which depend on it, and a failed submission cancels the jobs submitted
# before it. Every job is charged to the Slurm account A if given, or to the
# default account otherwise. With --resume, DIR's programs are kept rather
# than generated, only the variants not yet timed are timed, and the jobs are
# added to DIR/jobs.txt. Each chunk is timed for --hours H, by default 6, and
# stops then, since its programs cost too differently to choose how many fill
# the time, so more are generated than any chunk could time. Options of this
# script come before DIR, and those after it are generate_physical_data's.
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
# The steps run from copies of the repository elsewhere, so paths are absolute.
absolute_model() {
  local path=${1#*=}
  echo "${1%%=*}=$(cd -- "$(dirname -- "$path")" && pwd)/$(basename -- "$path")"
}
while [[ $# -gt 0 ]]; do
  case $1 in
    --account) account=$2; shift 2 ;;
    --account=*) account=${1#*=}; shift ;;
    --stage) stage=$2; shift 2 ;;
    --stage=*) stage=${1#*=}; shift ;;
    --model) models+=(--model "$(absolute_model "$2")"); shift 2 ;;
    --model=*) models+=(--model "$(absolute_model "${1#*=}")"); shift ;;
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
for basis in "$stage" "${bases[@]}"; do
  if [[ " ${unique[*]} " != *" $basis "* ]]; then
    unique+=("$basis")
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

# If a submission fails, cancel the jobs submitted before it, rather than
# leaving part of a run queued, and record only the jobs of a whole run.
submitted=()
cancel_submitted() {
  if [[ $? -ne 0 && ${#submitted[@]} -gt 0 ]]; then
    echo "$0: submission failed; cancelling ${submitted[*]}" >&2
    scancel "${submitted[@]}" || true
  fi
}
trap cancel_submitted EXIT
# A command substitution doesn't inherit set -e, so fail explicitly, and on an
# empty ID, which sbatch can print while failing.
submit() {
  local id
  id=$(sbatch --parsable --chdir="$repo_directory" --kill-on-invalid-dep=yes \
    ${account:+"--account=$account"} "$@") || return 1
  id=${id%%;*}
  [[ -n $id ]] || return 1
  echo "$id"
}
# submit runs in a subshell, so the jobs are recorded here, as NAME ID.
records=()
record() {
  submitted+=("$2")
  records+=("$1 $2")
}

jobs=$directory/jobs.txt
array=--array=0-$((chunks - 1))
logs=$directory/logs
generate=
step=scripts/fitting/step.slurm
if [[ ! $resume ]]; then
  generate=$(submit "$array" --job-name=generate --cpus-per-task=8 --mem=64G \
    --time=02:00:00 --output="$logs/generate-%A_%a.log" \
    "$step" generate_physical_data "$directory" --stage "$stage" "$@")
  record generate "$generate"
fi
# Each timing task takes a whole node, with all of its memory, so other jobs
# don't disturb it and large inputs and outputs fit. A task stops starting
# variants after its hours, and the limit leaves two more for installing and
# starting Julia and finishing the last variant, which can run for minutes.
limit=$((hours + 2)):00:00
time=$(submit "$array" ${generate:+"--dependency=aftercorr:$generate"} \
  --job-name=time --exclusive --mem=0 --time="$limit" \
  --output="$logs/time-%A_%a.log" \
  "$step" time_physical_data "$directory" "${models[@]}" --repeat "$repeat" \
  --hours "$hours" $resume)
record time "$time"
coalesce=$(submit --dependency="afterany:$time" --job-name=coalesce \
  --cpus-per-task=1 --mem=16G --time=01:00:00 \
  --output="$logs/coalesce-%j.log" \
  "$step" coalesce_physical_data "$directory")
record coalesce "$coalesce"
# Fitting featurizes the examples as they were timed, with the same models.
for basis in "${bases[@]}"; do
  fit=$(submit --dependency="afterok:$coalesce" --job-name="fit-$basis" \
    --cpus-per-task=1 --mem=16G --time=02:00:00 --mail-type=END,FAIL \
    --output="$logs/fit-$basis-%j.log" \
    "$step" fit_physical_data "$directory/examples.json" --basis "$basis" \
    "${models[@]}" "$directory/$basis.model.json")
  record "fit-$basis" "$fit"
done

[[ $resume ]] || : >"$jobs"
printf '%s\n' "${records[@]}" >>"$jobs"
echo "submitted $chunks chunks of the $stage stage:"
printf '  %s\n' "${records[@]}"
echo "check progress with: scripts/fitting/status_physical_data.sh $directory"
