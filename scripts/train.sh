#!/usr/bin/env bash
set -euo pipefail
python -m dpies.training.train \
  --config configs/train.yaml \
  --cache-dir ./cache/train \
  --val-cache-dir ./cache/val \
  --output-dir ./runs/dpies_main
