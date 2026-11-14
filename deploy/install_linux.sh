#!/usr/bin/env bash
set -euo pipefail

# Native Linux installer for Car Tracker (no Docker required)
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVICE_NAME="car-tracker"
CURRENT_USER="${SUDO_USER:-$USER}"

echo "==> Installing system dependencies (ffmpeg, libgl1, curl)..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq ffmpeg libgl1 libglib2.0-0 curl
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "==> Installing uv..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

UV_BIN="$(command -v uv)"

echo "==> Syncing Python 3.12 environment in $APP_DIR..."
cd "$APP_DIR"
"$UV_BIN" python install 3.12
"$UV_BIN" sync --python 3.12

if [ ! -f "$APP_DIR/.env" ]; then
  cp "$APP_DIR/.env.example" "$APP_DIR/.env"
  echo "==> Created .env from .env.example (edit .env or configure via Web UI)."
fi

echo "==> Installing systemd service ($SERVICE_NAME)..."
TMP_UNIT="$(mktemp)"
sed \
  -e "s|__USER__|$CURRENT_USER|g" \
  -e "s|__APP_DIR__|$APP_DIR|g" \
  -e "s|__UV_BIN__|$UV_BIN|g" \
  "$APP_DIR/deploy/car-tracker.service" > "$TMP_UNIT"

sudo mv "$TMP_UNIT" "/etc/systemd/system/${SERVICE_NAME}.service"
sudo systemctl daemon-reload
sudo systemctl enable "$SERVICE_NAME"
sudo systemctl restart "$SERVICE_NAME"

echo "==> Done! Check status with: sudo systemctl status $SERVICE_NAME"
echo "==> View logs with: journalctl -u $SERVICE_NAME -f"
