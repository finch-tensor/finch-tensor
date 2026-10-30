#!/usr/bin/env bash
# Fit the formatter's cost model to every program of DIR which has been timed
# and featurized. Submit from the repository, or run every step with
# scripts/launch_format_data.sh.
#
#     sbatch scripts/fit_format_data.slurm DIR [model.json]
#SBATCH --job-name=fit-format-data
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=fit-format-data-%j.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahrens@gatech.edu

set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

pixi run --frozen -e fitting python -m scripts.fit_format_data "$@"
