#!/usr/bin/env bash
set -euo pipefail

APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_USER="${SUDO_USER:-$USER}"
UV_BIN="$(command -v uv || true)"

if [[ -z "$UV_BIN" ]]; then
  for candidate in "$HOME/.local/bin/uv" "/home/$RUN_USER/.local/bin/uv" "/usr/local/bin/uv"; do
    if [[ -x "$candidate" ]]; then
      UV_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$UV_BIN" ]]; then
  echo "uv not found. Install it first: curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

echo "Syncing Python dependencies with $UV_BIN in $APP_DIR..."
(cd "$APP_DIR" && "$UV_BIN" sync --no-dev)

SERVICE_TMP="$(mktemp)"
sed \
  -e "s|__USER__|$RUN_USER|g" \
  -e "s|__APP_DIR__|$APP_DIR|g" \
  -e "s|__UV_BIN__|$UV_BIN|g" \
  "$APP_DIR/deploy/car-tracker.service" > "$SERVICE_TMP"

echo "Installing /etc/systemd/system/car-tracker.service (requires sudo)..."
sudo cp "$SERVICE_TMP" /etc/systemd/system/car-tracker.service
rm -f "$SERVICE_TMP"

sudo systemctl daemon-reload
sudo systemctl enable --now car-tracker
sudo systemctl status car-tracker --no-pager
