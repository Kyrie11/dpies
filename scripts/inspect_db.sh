#!/usr/bin/env bash
set -euo pipefail
DATA_ROOT=${DATA_ROOT:-/data0/senzeyu2/dataset/nuplan/data/cache}
python -m dpies.data.schema_probe --data-root "$DATA_ROOT" --limit 1
