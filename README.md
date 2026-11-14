# Reolink Car & Red-Zone Illegal Parking Tracker

Real-time vehicle & person tracker, traffic hotspot heatmap, and red-zone illegal parking detector for your **Reolink Lumos Pro** connected to a **Reolink NVR** (`192.168.68.51`, Channel `07`).

## Quick Start (macOS)

1. Install dependencies with `uv` (Python 3.12):
   ```bash
   uv sync --extra dev --python 3.12
   ```
2. Copy `.env.example` to `.env` (optional — you can also configure everything in the Web UI):
   ```bash
   cp .env.example .env
   ```
3. Start the server:
   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```
4. Open `http://localhost:8000` in your browser:
   - Click **Settings**, enter your Reolink NVR username and password, and click **Test Connection**.
   - Switch between **Main Stream (4K)** and **Sub Stream**, and adjust **Processing FPS** (`2–5 FPS` recommended) and **YOLO Inference Size** (`640`, `960`, or `1280`).
   - Click **+ Draw Red Zone** to outline your red curb on the live video feed.

## Linux Server Deployment (`192.168.68.89`)

Copy this folder to your Linux server and run the native installer (no Docker needed):

```bash
chmod +x deploy/install_linux.sh
./deploy/install_linux.sh
```

Manage the background service:
- `sudo systemctl status car-tracker`
- `sudo systemctl restart car-tracker`
- `journalctl -u car-tracker -f`

## Running Tests

```bash
uv run pytest
```
