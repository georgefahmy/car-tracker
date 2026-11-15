#!/usr/bin/env bash
set -euo pipefail

export PYTHONUNBUFFERED=1
export DATA_DIR=/data

if [ -f /data/options.json ]; then
  TZ_OPT=$(jq -r '.tz // empty' /data/options.json || true)
  if [ -n "${TZ_OPT}" ]; then
    export TZ="${TZ_OPT}"
  fi
fi

mkdir -p /data/snapshots
cd /app
exec uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --no-access-log
