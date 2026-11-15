# Car & Red-Zone Tracker (Home Assistant Add-on)

## Installation
1. In Home Assistant, go to **Settings → Add-ons → Add-on Store**.
2. Click the **⋮** menu (top right) → **Repositories**.
3. Add `https://github.com/georgefahmy/car-tracker`.
4. Find **Car & Red-Zone Tracker** in the store and click **Install**.
5. Click **Start**, then click **Open Web UI** (accessible at `http://<home-assistant-ip>:8000`).

## Data Persistence
- All settings, zones, traffic lines, 3D ground calibration, violation history, and snapshots are stored in `/data` (`tracker.db` and `snapshots/`), which Home Assistant Supervisor persists and includes in backups automatically.
