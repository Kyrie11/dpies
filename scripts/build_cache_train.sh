#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-/data0/senzeyu2/dataset/nuplan/data/cache}
MAP_ROOT=${MAP_ROOT:-/data0/senzeyu2/dataset/nuplan/maps}
OUT=${OUT:-./cache/train}
python -m dpies.data.preprocess_nuplan \
  --data-root "$DATA_ROOT" \
  --map-root "$MAP_ROOT" \
  --output-dir "$OUT" \
  --subdirs train_boston train_singapore train_pittsburgh train_vegas_2 \
  --sample-interval-s 1.0 \
  --history-seconds 2.0 \
  --future-seconds 8.0 \
  --dt 0.5 \
  --max-agents 64 \
  --max-actions 32 \
  --max-evidence-units 128 \
  --continue-on-error
