#!/usr/bin/env bash
# Submit every step of fitting the cost model of the stage STAGE of
# scripts/fitting/physical_bases.py, by default protocol, to DIR, with the
# models of the stages below it given as --model NAME=PATH, or the built-in
# ones. Each of --chunks N chunks, by default 64, is generated, then timed for
# --hours H, by default 1, since its programs cost too differently to choose
# how many fill the time. Once every chunk ends, however it ends, the timed
# programs are coalesced into DIR/examples.json, which is small enough to fit
# again anywhere, and the model of the stage is fitted to them and written to
# DIR/STAGE.model.json, which the next stage takes as its --model. A failed
# step cancels the steps after it. Programs and timings go to DIR/data, logs
# to DIR/logs, and job IDs to DIR/jobs.txt, for status_physical_data.sh. With
# --resume, DIR's programs are kept rather than generated, and the jobs are
# added to DIR/jobs.txt. Options before DIR are this script's; those after it
# are those of the generator of the stage, generate_protocol_data for the
# protocol stage, and generate_loop_data and generate_factorize_data for the
# loop and factorize stages.
#
# --stage may be given more than once, for stages which each fit a model
# that later ones use, such as --stage protocol --stage loop --stage factorize,
# the models the physical optimizers use. Each stage then goes to DIR/STAGE,
# and is generated at once, but timed only once the stages whose models it
# chooses with have fitted them, which it takes as its --model, over those
# given: the loop stage chooses with the protocol model, and the factorize
# stage with that and the loop model. The loop fit tunes the
# region cutoff of its classes of writes. The generator options are given to
# each stage.
set -euo pipefail

usage="usage: $0 [--account A] [--stage STAGE]... [--model NAME=PATH]... \
[--chunks N] [--hours H] [--resume] DIR [generator options]"
account=
stages=()
models=()
chunks=64
hours=1
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
    --stage) stages+=("$2"); shift 2 ;;
    --stage=*) stages+=("${1#*=}"); shift ;;
    --model) models+=(--model "$(absolute_model "$2")"); shift 2 ;;
    --model=*) models+=(--model "$(absolute_model "${1#*=}")"); shift ;;
    --resume) resume=1; shift ;;
    --chunks) chunks=$2; shift 2 ;;
    --hours) hours=$2; shift 2 ;;
    -h | --help) echo "$usage"; exit 0 ;;
    *) break ;;
  esac
done
if [[ $# -lt 1 ]]; then
  echo "$usage" >&2
  exit 2
fi
[[ ${#stages[@]} -gt 0 ]] || stages=(protocol)
mkdir -p "$1"
root=$(cd -- "$1" && pwd)
shift
if [[ $resume && $# -gt 0 ]]; then
  echo "$0: --resume generates nothing, so takes no generate options: $*" >&2
  exit 2
fi
# Each stage's directory: DIR itself for one stage, and DIR/STAGE for several.
directories=()
for stage in "${stages[@]}"; do
  directory=$root
  [[ ${#stages[@]} -eq 1 ]] || directory=$root/$stage
  mkdir -p "$directory/logs"
  # Generating again would replace programs whose timings and features remain.
  if [[ ! $resume ]] && compgen -G "$directory/data/*.program.json" >/dev/null; then
    echo "$0: $directory already has programs; pass --resume before DIR" >&2
    exit 2
  fi
  directories+=("$directory")
done

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

array=--array=0-$((chunks - 1))
step=scripts/fitting/step.slurm
# Each timing task takes a whole node, with all of its memory, so other jobs
# don't disturb it and large inputs and outputs fit. A task stops starting
# variants after its hours, and the limit leaves two more for installing and
# starting Julia and finishing the last variant, which can run for minutes.
limit=$((hours + 2)):00:00
# submit runs in a subshell, so the jobs are recorded here, as NAME ID.
records=()
record() {
  submitted+=("$2")
  records+=("$1 $2")
}
# The stages whose models the timing of a stage chooses with, as resolve in
# physical_bases.py does. The protocol stage gives every choice.
uses() {
  case $1 in
    loop) echo protocol ;;
    factorize) echo protocol loop ;;
  esac
}
# The fit job and the model of each stage submitted so far.
declare -A fits fitted_models
summary=()
for n in "${!stages[@]}"; do
  stage=${stages[$n]}
  directory=${directories[$n]}
  logs=$directory/logs
  records=()
  dependencies=()
  stage_models=("${models[@]}")
  for used in $(uses "$stage"); do
    if [[ -n ${fits[$used]:-} ]]; then
      dependencies+=("afterok:${fits[$used]}")
      stage_models+=(--model "$used=${fitted_models[$used]}")
    elif [[ " ${stages[*]} " == *" $used "* ]]; then
      echo "$0: the $stage stage uses the $used model, so --stage $used goes first" >&2
      exit 2
    fi
  done
  case $stage in
    loop) generator=generate_loop_data ;;
    factorize) generator=generate_factorize_data ;;
    *) generator=generate_protocol_data ;;
  esac
  generate=
  if [[ ! $resume ]]; then
    generate=$(submit "$array" --job-name="generate-$stage" --cpus-per-task=8 \
      --mem=64G --time=02:00:00 --output="$logs/generate-%A_%a.log" \
      "$step" "$generator" "$directory" --stage "$stage" "$@")
    record generate "$generate"
  fi
  [[ -z $generate ]] || dependencies+=("aftercorr:$generate")
  dependency=$(IFS=,; echo "${dependencies[*]}")
  time=$(submit "$array" ${dependency:+"--dependency=$dependency"} \
    --job-name="time-$stage" --exclusive --mem=0 --time="$limit" \
    --output="$logs/time-%A_%a.log" \
    "$step" time_physical_data "$directory" "${stage_models[@]}" --hours "$hours")
  record time "$time"
  # The chunks which ran out of time still count, so coalescing waits for the
  # timing to end however it ends.
  coalesce=$(submit --dependency="afterany:$time" --job-name="coalesce-$stage" \
    --cpus-per-task=1 --mem=16G --time=01:00:00 \
    --output="$logs/coalesce-%j.log" \
    "$step" coalesce_physical_data "$directory")
  record coalesce "$coalesce"
  # Fitting featurizes the examples as they were timed, with the same models.
  tune=()
  [[ $stage != loop ]] || tune=(--tune-cutoff)
  fit=$(submit --dependency="afterok:$coalesce" --job-name="fit-$stage" \
    --cpus-per-task=1 --mem=16G --time=02:00:00 --mail-type=END,FAIL \
    --output="$logs/fit-%j.log" \
    "$step" fit_physical_data "$directory/examples.json" --basis "$stage" \
    "${tune[@]}" "${stage_models[@]}" "$directory/$stage.model.json")
  record fit "$fit"
  fits[$stage]=$fit
  fitted_models[$stage]=$directory/$stage.model.json

  [[ $resume ]] || : >"$directory/jobs.txt"
  printf '%s\n' "${records[@]}" >>"$directory/jobs.txt"
  summary+=("$stage stage, in $directory:" "${records[@]/#/  }")
done

echo "submitted $chunks chunks of each stage:"
printf '  %s\n' "${summary[@]}"
echo "check progress with: scripts/fitting/status_physical_data.sh $root"
