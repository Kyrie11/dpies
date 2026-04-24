#!/usr/bin/env bash
set -euo pipefail
python -m dpies.evaluation.evaluate \
  --config configs/eval.yaml \
  --checkpoint ./runs/dpies_main/best.pt \
  --cache-dir ./cache/val \
  --output-dir ./runs/dpies_main/eval
