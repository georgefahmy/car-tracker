# Changelog

## 1.0.12
- **Native Home Assistant Entities & Events (#1)**: Automatically publishes `sensor.car_tracker_vehicles_today`, `sensor.car_tracker_people_today`, `sensor.car_tracker_active_vehicles`, `sensor.car_tracker_last_speed`, `sensor.car_tracker_avg_speed`, `sensor.car_tracker_violations_total`, `sensor.car_tracker_last_plate`, and `binary_sensor.car_tracker_red_zone_occupied` via Supervisor Core API (`homeassistant_api: true`), plus fires `car_tracker_violation` and `car_tracker_speeding` automation events.
- **On-Demand License Plate Recognition (#2)**: Uses local ONNX OCR (`rapidocr-onnxruntime`) only when a Red-Zone violation or Speeding snapshot is captured (`0%` live-stream CPU overhead), storing plates in SQLite, showing plate badges & filtering in Violations, exporting in CSV, appending to Discord/webhook alerts, and updating `sensor.car_tracker_last_plate`.
- **ONNX Runtime + Intel iGPU Acceleration (#5)**: Adds `onnx` (ONNX Runtime CPU with `allow_spinning="0"` and capped intra-op threads) and `openvino_gpu` (`intel:gpu` offloading via `/dev/dri` and `video: true`), falling back cleanly if no iGPU is exposed.

## 1.0.11
- Optimized the web UI for mobile phones and Home Assistant Companion / Ingress views: 2-column KPI grid, 44px coarse-pointer touch targets, safe-area inset & `dvh` bottom-sheet modals, larger touch hit radii for dragging zone corners, tap-to-inspect tooltips on Hourly Traffic, and iOS input auto-zoom prevention.

## 1.0.10
- Automatically resets all counts (vehicles, people, hourly traffic, line crossings, and speed history) after a daily report is successfully sent to the webhook.

## 1.0.9
- Enabled Home Assistant **Ingress** (`Show in sidebar`) so you can open Car Tracker directly inside the Home Assistant left sidebar menu (`Car Tracker`).
- Converted frontend asset, snapshot, and API URLs to relative paths so both Ingress and direct port `8000` access work seamlessly.

## 1.0.8
- Added high-res snapshot capture & Discord/webhook alerting when a vehicle exceeds the configurable Speed Limit (default `25 mph`) between two traffic lines.

## 1.0.7
- Switched `Auto` inference backend back to PyTorch CPU (OpenVINO's oneTBB runtime caused 75–100% CPU spin-waiting vs ~42% on PyTorch; OpenVINO remains selectable manually in Settings).
- Skipped preview annotation & JPEG encoding whenever no browser viewer has polled in the last 5 seconds, and capped active preview encoding to ~12 FPS instead of 25 FPS.

## 1.0.6
- Fixed 100% CPU spin-waiting and thread contention under OpenVINO/PyTorch by setting `OMP_WAIT_POLICY=PASSIVE`, `KMP_BLOCKTIME=0`, `OMP_NUM_THREADS=4`, `MKL_NUM_THREADS=4`, and capping OpenCV/PyTorch worker threads.

## 1.0.5
- Fixed Docker build cache in the HA add-on `Dockerfile` (`COPY config.yaml` before `git clone`) so version bumps always pull the latest app code, OpenVINO backend, and dependencies from `main`.

## 1.0.4
- Disabled noisy per-request HTTP access logs (`--no-access-log` in Uvicorn) so rapid `/api/stream/frame.jpg` and `/api/stats` polls no longer flood the add-on logs.

## 1.0.3
- Added automatic **OpenVINO** export & caching on x86_64 Linux (Beelink/Intel mini-PCs) for 2–3× faster CPU inference.
- Removed the 4-thread CPU cap so PyTorch/OpenVINO can utilize all available CPU cores.
- Added **YOLO Model** (`yolov8n.pt`, `yolo11n.pt`, `yolo11s.pt`, `yolov8s.pt`) and **Inference Backend** (`auto`, `openvino`, `pytorch`) selectors in Settings, plus live backend status in the header.

## 1.0.2
- Added `CHANGELOG.md` so Home Assistant displays release notes in the add-on Changelog tab and update dialog.

## 1.0.1
- Fixed Home Assistant Supervisor Docker build by pinning `FROM python:3.12-slim` directly in `Dockerfile`.

## 1.0.0
- Initial Home Assistant OS add-on release with YOLO vehicle/person tracking, red-zone violation detection, 3D ground calibration, auto red-curb detection, two-line speed estimation (with stationary speed clearing), and Discord alerts/daily reports.
