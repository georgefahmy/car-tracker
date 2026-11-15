# Car & Red-Zone Tracker (Home Assistant Add-on)

## Installation
1. In Home Assistant, go to **Settings → Add-ons → Add-on Store**.
2. Click the **⋮** menu (top right) → **Repositories**.
3. Add `https://github.com/georgefahmy/car-tracker`.
4. Find **Car & Red-Zone Tracker** in the store and click **Install**.
5. Click **Start**, then click **Car Tracker** in the Home Assistant sidebar (Ingress) or **Open Web UI** (accessible at `http://<home-assistant-ip>:8000`).

## Native Home Assistant Entities & Events
When running inside Home Assistant OS (`homeassistant_api: true`), the add-on automatically publishes live states and triggers automatable bus events:

### Auto-Created Entities
- `sensor.car_tracker_vehicles_today` — total tracked vehicles (with `unique_vehicles` & `active_vehicles` attributes)
- `sensor.car_tracker_people_today` — total tracked people (with `unique_people` & `active_people` attributes)
- `sensor.car_tracker_active_vehicles` — live vehicles currently in frame
- `sensor.car_tracker_last_speed` — most recent 2-line speed measurement (`mph`)
- `sensor.car_tracker_avg_speed` — rolling average speed (`mph`)
- `sensor.car_tracker_violations_total` — logged red-zone & speeding violation count
- `sensor.car_tracker_last_plate` — most recently read license plate from violation/speeding snapshots
- `binary_sensor.car_tracker_red_zone_occupied` — `on` whenever a vehicle is dwelling in a red zone

### Bus Events (for Automations)
- `car_tracker_violation` — fired on illegal red-zone parking detection (includes `zone_name`, `vehicle_class`, `dwell_seconds`, `license_plate`, `snapshot_filename`)
- `car_tracker_speeding` — fired when a vehicle exceeds the Speed Limit (includes `speed_mph`, `vehicle_class`, `license_plate`, `snapshot_filename`)

## Performance Backends (Beelink / Intel x86)
- **ONNX Runtime (`onnx`)**: Fast CPU inference with thread spin-waiting disabled for low idle CPU.
- **OpenVINO Intel iGPU (`openvino_gpu`)**: Offloads YOLO inference onto Intel integrated graphics (`/dev/dri`) when available, falling back cleanly to ONNX/PyTorch if no iGPU is exposed.

## Data Persistence
- All settings, zones, traffic lines, 3D ground calibration, violation history, license plates, and snapshots are stored in `/data` (`tracker.db` and `snapshots/`), which Home Assistant Supervisor persists and includes in backups automatically.
