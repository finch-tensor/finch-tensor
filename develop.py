#!/usr/bin/env poetry run python
import argparse
import os
import shutil

import juliapkg

script_dir = os.path.dirname(os.path.abspath(__file__))
source_file = os.path.join(script_dir, "src/finch/juliapkg.json")
backup_file = os.path.join(script_dir, "src/finch/juliapkg.json.orig")

# Parse command-line arguments
usage = """
Usage:
    develop.py [--restore] [--path <path>]

Options:
    --restore   Restore the original juliapkg.json file.
    --path      Path to the local copy of Finch.jl [default: ../Finch.jl].
"""
parser = argparse.ArgumentParser(
    description=(
        "Development script for Finch. This script allows you to specify "
        "the location of a local copy of Finch.jl."
    ),
    usage=usage,
)
parser.add_argument(
    "--path",
    default=os.path.join(script_dir, "../Finch.jl"),
    help="Path to the Finch.jl package.",
)
parser.add_argument(
    "--restore", action="store_true", help="Restore the original juliapkg.json file."
)
args = parser.parse_args()

# Handle the --restore flag
if args.restore:
    try:
        shutil.copy(backup_file, source_file)
        print("Restored src/finch/juliapkg.json from backup.")
    except FileNotFoundError:
        print("Error: Backup file src/finch/juliapkg.json.orig does not exist.")
    except (OSError, PermissionError) as e:
        print(f"An error occurred: {e}")
    exit()

# Set the Finch path
finch_path = os.path.abspath(args.path)

# Define source and destination file paths and copy the file
try:
    if not os.path.exists(backup_file):
        shutil.copy(source_file, backup_file)
except (OSError, PermissionError) as e:
    print(f"An error occurred: {e}")

# Checkout Finch for development

juliapkg.rm("Finch", target="src/finch/juliapkg.json")
juliapkg.add(
    "Finch",
    "9177782c-1635-4eb9-9bfb-d9dfa25e6bce",
    dev=True,
    path=finch_path,
    target="src/finch/juliapkg.json",
)
