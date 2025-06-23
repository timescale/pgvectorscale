#!/bin/bash

# Helper script to run the issue 193 reproduction case

cd "$(dirname "$0")" || exit

# shellcheck disable=SC1091
source venv/bin/activate

echo "Cleaning up previous data..."
psql -h localhost -p 28817 -d postgres -c "DROP TABLE IF EXISTS embeddings CASCADE;" > /dev/null 2>&1

echo "Running reproduction script with parameters: $*"
python issue_193_repro.py "$@"