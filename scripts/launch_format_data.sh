#!/usr/bin/env bash
# Submit every step of fitting the formatter's cost model to DIR. Each chunk is
# timed and featurized as soon as it is generated, and the model is fitted to
# all of them once they are done, and written to DIR/model.json. Logs go to
# DIR/logs. A failed step cancels the steps which depend on it. Every job is
# charged to the Slurm account A if given, or to the default account otherwise.
#
#     scripts/launch_format_data.sh [--account A] [--chunks N] [--repeat N] DIR \
#         [generate_format_data options]
set -euo pipefail

usage="usage: $0 [--account A] [--chunks N] [--repeat N] DIR \
[generate_format_data options]"
account=
chunks=8
repeat=5
while [[ $# -gt 0 ]]; do
  case $1 in
    --account) account=$2; shift 2 ;;
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

cd -- "$(dirname -- "$0")"
repo_directory=$(git rev-parse --show-toplevel)
cd -- "$repo_directory"

# Install the environment and Julia's packages here, once, rather than in every
# task at once.
pixi install --frozen -e fitting
pixi run --frozen -e fitting python -c \
  "from finch.compile_jl.julia import init_julia; init_julia()"

submit() {
  local id
  id=$(sbatch --parsable --chdir="$repo_directory" --kill-on-invalid-dep=yes \
    ${account:+"--account=$account"} "$@")
  echo "${id%%;*}"
}

array=--array=0-$((chunks - 1))
logs=$directory/logs
generate=$(submit "$array" --output="$logs/generate-%A_%a.log" \
  scripts/generate_format_data.slurm "$directory" "$@")
time=$(submit "$array" --dependency="aftercorr:$generate" \
  --output="$logs/time-%A_%a.log" \
  scripts/time_format_data.slurm "$directory" --repeat "$repeat")
featurize=$(submit "$array" --dependency="aftercorr:$generate" \
  --output="$logs/featurize-%A_%a.log" \
  scripts/featurize_format_data.slurm "$directory")
fit=$(submit --dependency="afterok:$time:$featurize" \
  --output="$logs/fit-%j.log" \
  scripts/fit_format_data.slurm "$directory" "$directory/model.json")

echo "generate $generate, time $time, featurize $featurize, fit $fit"
echo "logs in $logs, model in $directory/model.json"
