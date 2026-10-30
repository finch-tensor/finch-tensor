#!/usr/bin/env bash
# Competition array for CPU frameworks. Shared logic: scripts/competition.sh.
# Submit together with the GPU array using scripts/submit-competition.sh.
#SBATCH --job-name=saps-competition-cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --mem=160G
#SBATCH --tmp=50G
#SBATCH --time=08:00:00
#SBATCH --array=0-63
#SBATCH --output=competition-cpu-%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahrens@gatech.edu

set -euo pipefail

# Slurm runs a spooled copy of this script, so find the repository from the
# submission directory rather than from $0.
repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi
repo_directory=$(git -C "$repo_directory" rev-parse --show-toplevel)

competition_device=cpu
source "$repo_directory/scripts/competition.sh"
