#!/usr/bin/env bash
# Show the progress of fitting the formatter's cost model to DIR: how many
# programs each step has finished, the states of the Slurm jobs which
# scripts/launch_format_data.sh submitted, the latest progress of each task
# which hasn't finished, and the logs which report errors.
#
#     scripts/status_format_data.sh DIR
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 DIR" >&2
  exit 2
fi
directory=$1

count() {
  find "$directory/data" -maxdepth 1 -name "*.$1" 2>/dev/null | wc -l | tr -d ' '
}
echo "programs: $(count program.json) generated, $(count timing.json) timed," \
  "$(count features.json) featurized"
if [[ -f $directory/model.json ]]; then
  echo "model: $directory/model.json"
else
  echo "model: not fitted yet"
fi

if [[ -f $directory/jobs.txt ]] && command -v sacct >/dev/null; then
  echo "jobs:"
  while read -r step id; do
    states=$(sacct -j "$id" -X -n -P -o State | sort | uniq -c |
      sed 's/^ *//' | paste -sd ',' - | sed 's/,/, /g')
    echo "  $step $id: ${states:-not started}"
  done <"$directory/jobs.txt"
fi

# Each step prints [i/n] as it finishes a program.
running=""
errors=""
for log in "$directory"/logs/*.log; do
  [[ -e $log ]] || continue
  progress=$(awk -F'[][/]' '/^\[[0-9]+\/[0-9]+\]/ {line = $0; i = $2; n = $3}
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
