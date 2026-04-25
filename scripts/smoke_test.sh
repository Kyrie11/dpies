#!/usr/bin/env bash
set -euo pipefail
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
rm -rf /tmp/dpies_smoke
python -m dpies.tools.make_toy_cache --output-dir /tmp/dpies_smoke/train --num-samples 16
python -m dpies.tools.make_toy_cache --output-dir /tmp/dpies_smoke/val --num-samples 8
python -m dpies.training.train \
  --config configs/train.yaml \
  --cache-dir /tmp/dpies_smoke/train \
  --val-cache-dir /tmp/dpies_smoke/val \
  --output-dir /tmp/dpies_smoke/run \
  --override training.epochs=1 data.batch_size=2 data.num_workers=0 model.hidden_dim=64 model.pair_chunk_size=32
python -m dpies.evaluation.evaluate \
  --checkpoint /tmp/dpies_smoke/run/best.pt \
  --cache-dir /tmp/dpies_smoke/val \
  --output-dir /tmp/dpies_smoke/eval
