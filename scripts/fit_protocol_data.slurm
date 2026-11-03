#!/usr/bin/env bash
# Fit the protocol cost model to the programs which
# scripts/condense_protocol_data.slurm collected into DATA. Submit from the
# repository, or run every step with scripts/launch_protocol_data.sh.
#
#     sbatch scripts/fit_protocol_data.slurm DATA [model.json]
#SBATCH --job-name=fit-protocol-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=fit-protocol-data-%j.log
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pixi run --frozen -e fitting python -m scripts.fit_protocol_data "$@"
