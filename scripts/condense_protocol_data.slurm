#!/usr/bin/env bash
# Collect the features of one basis and the timings of the programs of DIR
# into one JSON file. Submit from the repository, or run every step with
# scripts/launch_protocol_data.sh.
#
#     sbatch scripts/condense_protocol_data.slurm DIR [--basis NAME] [data.json]
#SBATCH --job-name=condense-protocol-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=condense-protocol-data-%j.log
#SBATCH --mail-type=FAIL

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pixi run --frozen -e fitting python -m scripts.condense_protocol_data "$@"
