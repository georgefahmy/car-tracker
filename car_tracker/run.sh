#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export DATA_DIR=/data
export OMP_WAIT_POLICY=PASSIVE
export KMP_BLOCKTIME=0
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export LIBVA_DRIVER_NAME=iHD
export NEO_CACHE_PERSISTENT=1
export NEO_CACHE_DIR=/data/models/neo_cache

if [ -f /data/options.json ]; then
  TZ_OPT=$(jq -r '.tz // empty' /data/options.json || true)
  if [ -n "${TZ_OPT}" ]; then
    export TZ="${TZ_OPT}"
  fi
fi

mkdir -p /data/snapshots /data/models/neo_cache /data/models/ov_cache
cd /app
exec uv run --no-sync uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log
