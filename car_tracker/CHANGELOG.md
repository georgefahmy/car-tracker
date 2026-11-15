# Changelog

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
