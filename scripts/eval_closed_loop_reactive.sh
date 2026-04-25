#!/usr/bin/env bash
set -euo pipefail
: "${DPIES_CHECKPOINT:?Set DPIES_CHECKPOINT to the trained DPIES checkpoint path}"
CHECKPOINT="${DPIES_CHECKPOINT}" LIMIT_SCENARIOS="${LIMIT_TOTAL_SCENARIOS:-${LIMIT_SCENARIOS:-20}}" bash scripts/run_closed_loop_reactive_idm.sh
