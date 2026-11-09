#!/usr/bin/env bash
# Show the progress of fitting the cost model of a stage to DIR: how many
# programs are generated, timed, and coalesced, the models fitted, the
# states of the Slurm jobs which scripts/fitting/launch_physical_data.sh
# submitted, the latest progress of each task which hasn't finished, and the
# logs which report errors. For a DIR of several stages, which
# launch_physical_data.sh puts in DIR/STAGE, the progress of each.
#
#     scripts/fitting/status_physical_data.sh DIR
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DIR" >&2
  exit 2
fi
directory=$1
if [[ ! -f $directory/jobs.txt ]] && compgen -G "$directory/*/jobs.txt" >/dev/null; then
  for jobs in "$directory"/*/jobs.txt; do
    echo "== $(basename "$(dirname "$jobs")")"
    "$0" "$(dirname "$jobs")"
  done
  exit 0
fi

count() {
  find "$directory/data" -maxdepth 1 -name "*.$1" 2>/dev/null | wc -l | tr -d ' '
}
echo "programs: $(count program.json) generated, $(count timing.json) timed"
# Each coalesced example is recorded under its name, and nothing else is.
if [[ -f $directory/examples.json ]]; then
  coalesced=$(grep -o '"name": "' "$directory/examples.json" | wc -l | tr -d ' ')
  echo "  $coalesced coalesced into examples.json"
fi
models=$(find "$directory" -maxdepth 1 -name "*.model.json" 2>/dev/null | sort)
if [[ -n $models ]]; then
  echo "models:"
  sed 's/^/  /' <<<"$models"
else
  echo "models: none fitted yet"
fi

if [[ -f $directory/jobs.txt ]] && command -v sacct >/dev/null; then
  echo "jobs:"
  while read -r step id; do
    states=$(sacct -j "$id" -X -n -P -o State | sort | uniq -c |
      sed 's/^ *//' | paste -sd ',' - | sed 's/,/, /g')
    echo "  $step $id: ${states:-not started}"
  done <"$directory/jobs.txt"
fi

# Each step prints [i/n] as it finishes a program, and stopped after its hours.
running=""
errors=""
for log in "$directory"/logs/*.log; do
  [[ -e $log ]] || continue
  progress=$(awk -F'[][/]' '/^\[[0-9]+\/[0-9]+\]/ {line = $0; i = $2; n = $3}
    /^stopped after/ {line = ""}
    END {if (line != "" && i != n) print line}' "$log")
  [[ -n $progress ]] && running+="  $(basename "$log"): $progress"$'\n'
  error=$(grep -E 'Traceback|Error|error:' "$log" | tail -1 || true)
  [[ -n $error ]] && errors+="  $(basename "$log"): $error"$'\n'
done
if [[ -n $running ]]; then
  printf "in progress:\n%s" "$running"
fi
if [[ -n $errors ]]; then
  printf "errors:\n%s" "$errors"
fi
