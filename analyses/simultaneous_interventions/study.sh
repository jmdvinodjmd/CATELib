#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

uv run python -m catelib.run --config analyses/simultaneous_interventions/config.json --method s --tune "$@"
uv run python -m catelib.run --config analyses/simultaneous_interventions/config.json --method ridge --tune "$@"

# Final tables and figures: saved results only, no additional training.
uv run python -m catelib.run --config analyses/simultaneous_interventions/config.json --report-only "$@"
