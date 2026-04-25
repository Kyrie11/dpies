#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-/data0/senzeyu2/dataset/nuplan/data/cache}
MAP_ROOT=${MAP_ROOT:-/data0/senzeyu2/dataset/nuplan/maps}
OUT=${OUT:-./cache/val}
EXTRA_ARGS=()
if [[ "${REQUIRE_MAP:-0}" == "1" ]]; then EXTRA_ARGS+=(--require-map); fi
if [[ "${DISABLE_MAP:-0}" == "1" ]]; then EXTRA_ARGS+=(--disable-map); fi
if [[ "${SKIP_EXISTING:-0}" == "1" ]]; then EXTRA_ARGS+=(--skip-existing); fi
if [[ "${CREATE_SQLITE_INDEXES:-0}" == "1" ]]; then EXTRA_ARGS+=(--create-sqlite-indexes); fi
if [[ "${USE_SCENARIO_API:-0}" == "1" ]]; then EXTRA_ARGS+=(--use-scenario-api); fi
if [[ -n "${SENSOR_ROOT:-}" ]]; then EXTRA_ARGS+=(--sensor-root "$SENSOR_ROOT"); fi
if [[ -n "${MAP_VERSION:-}" ]]; then EXTRA_ARGS+=(--map-version "$MAP_VERSION"); fi
python -m dpies.data.preprocess_nuplan \
  --data-root "$DATA_ROOT" \
  --map-root "$MAP_ROOT" \
  --output-dir "$OUT" \
  --subdirs val \
  --sample-interval-s "${SAMPLE_INTERVAL_S:-1.0}" \
  --history-seconds 2.0 \
  --future-seconds 8.0 \
  --dt 0.5 \
  --max-agents 64 \
  --max-actions 32 \
  --max-evidence-units 128 \
  --continue-on-error \
  "${EXTRA_ARGS[@]}"
