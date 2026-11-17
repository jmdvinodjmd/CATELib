#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

# One small binary dataset; each estimator has its own tuning search.
# METHOD=name optionally runs one method; H-learner is included normally.

if [[ -z "${METHOD:-}" || "${METHOD}" == "h" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method h --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "s" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method s --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "xh" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method xh --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "xs" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method xs --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "t" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method t --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "ridge" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method ridge --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "dec" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method dec --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "vsr" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method vsr --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "scp" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method scp --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "scp-s" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method scp-s --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "cisi" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method cisi --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "cihsi" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method cihsi --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "ncore" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method ncore --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "tece" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method tece --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "tece-marginal" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method tece-marginal --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "dr" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method dr --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "tarnet" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method tarnet --tune "$@"
fi
if [[ -z "${METHOD:-}" || "${METHOD}" == "cfr" ]]; then
  uv run python -m catelib.run --config analyses/method_comparison/config.json --scenario binary --method cfr --tune "$@"
fi

# Final tables and figures: saved results only, no additional training.
report_options=()
if [[ -n "${METHOD:-}" ]]; then report_options+=(--method "$METHOD"); fi
uv run python -m catelib.run --config analyses/method_comparison/config.json --report-only "${report_options[@]}" "$@"
