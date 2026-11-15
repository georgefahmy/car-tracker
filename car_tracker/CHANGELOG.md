# Changelog

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
