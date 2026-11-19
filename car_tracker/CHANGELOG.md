# Changelog

## 1.0.37
- **High-Contrast Traffic Heatmap**: Replaced heavy logarithmic compression and concave opacity boosting with 99th-percentile peak scaling (guarded against sprawling low-traffic noise), a `ratio ** 0.75` color curve, and a convex `ratio ** 1.10` opacity curve at up to `80%` peak opacity. Heavy traffic corridors now blaze clearly in glowing amber/solar yellow (vehicles) or electric cyan/ice-aqua (people), while light and occasional paths fade softly into subtle low-opacity crimson/cobalt whispers without bloated halos.

## 1.0.36
- **Streamlined 6-Card KPI Bar**: Removed the duplicate **Unique Vehicles** card (keeping **Vehicles (Live)** and highlighting **Total Vehicles**) and updated **People (Live / Unique)** to **People (Live / Total)** with updated Reset Counts prompt/tooltip text, while preserving backend API and Home Assistant fields for compatibility.

## 1.0.35
- **Clean Offender-Only Violation, Speeding & Person Snapshots**: Updated saved snapshot rendering (`violation_*.jpg`, `speeding_*.jpg`, `person_*.jpg` for both the Web UI and Discord webhook alerts) to draw **only the offending vehicle/person's bounding box and label** on the clean camera frame — removing Red-Zone/Exclusion/Person-Zone polygon markings & fills, traffic/speed lines, heatmaps, bystander boxes, and anchor dots while keeping full overlays on the Live Camera Feed.

## 1.0.34
- **Separate Vehicle & Pedestrian Heatmaps with Distinct Color Ramps**: Split `HotspotTracker` into independent `vehicle_grid` and `person_grid` accumulators with separate `np.log1p` normalization and custom colorblind-safe OpenCV LUTs (**Warm Crimson → Amber → Gold** for vehicles; **Cool Sapphire → Electric Cyan → Ice-Aqua** for people) so heavy street car traffic never washes out pedestrian sidewalks.
- **1-Click Heatmap Layer Cycling (`All` / `Cars` / `People` / `OFF`)**: Clicking the top-bar `Heatmap` button now cycles cleanly through `OFF → All → Cars → People → OFF` with color-key swatches on the button, live preview/snapshot/Bird's-Eye View updates, and persisted `heatmap_mode` in SQLite.
- **2-Channel Disk Persistence (`/data/heatmap.npy`)**: Stores `[vehicle_grid, person_grid]` together on disk while automatically upgrading legacy single-channel 2D `.npy` files into `vehicle_grid`.

## 1.0.33
- **Final-Timestamped Red-Zone Discord / Webhook Alerts**: Deferred Red-Zone vehicle violation webhooks until the violation completes (or is flushed on shutdown, reset, source switch, or the 4-hour eviction cap), updating both the database record (`flagged_at`, `dwell_seconds`, `license_plate`) and the snapshot image from the vehicle's **last in-zone frame** before dispatching so Discord receives the end-of-violation camera timestamp and final `ILLEGAL PARK` dwell badge instead of the initial trigger frame.
- **Live In-Zone Snapshot Tracking & Human-Readable Webhook Durations**: Kept live Web UI cards, Home Assistant events, Speeding alerts, and after-hours Person security alerts immediate while tracking the latest in-zone base frame without overwriting higher-resolution 4K HTTP Snap stills or capturing empty-street frames after departure, and formatted Discord/webhook dwell durations with `format_dwell_duration` (`e.g. 2m 30s`) reading finalized JPEG bytes synchronously before background upload.

## 1.0.32
- **Persistent Cumulative Traffic Heatmap (Since Last Reset)**: Removed the exponential time decay (`decay = 1.0`) and added atomic disk persistence (`/data/heatmap.npy` + `/data/heatmap.meta.json`) saved every 30 seconds and on clean shutdown so accumulated traffic survives add-on restarts and updates.
- **Motion-Weighted Path Trails & Parked-Car Filtering**: Draws smooth frame-to-frame path segments along moving trajectories while capping stationary parked-car heat (and inheriting dwell state across ByteTrack ID flicker or immediately after a live reset) so parked vehicles never wash out moving street or sidewalk traffic.
- **Logarithmic Dynamic Range & Halo-Free Blending**: Switched heatmap normalization to `np.log1p` with intensity-proportional alpha blending (plus optional Bird's-Eye View heatmap projection) so occasional single-pass lanes and high-volume corridors stay readable together.
- **Count-Reset Synchronization & Accessible Action Hints**: Synchronized **Reset Counts** and the scheduled **Daily Summary Report** (`reset_after=True`) to clear the cumulative heatmap alongside counters while leaving **Clear Heatmap** independent, and upgraded top-bar action buttons to accessible `popover="hint"` + `interestfor` tooltips displaying the current accumulation start time.

## 1.0.31
- **Model & Backend CPU Impact Tiers**: Extended the unified Settings dropdown labels to `[Device · Backend · CPU: <Impact>]` (`Minimal`, `Very Low`, `Low`, `Moderate`, `High`, `Very High`), dynamically updating per-model CPU load and `Auto`/`PyTorch` host resolution whenever **Inference Backend** or hardware diagnostics change.

## 1.0.30
- **Unified Model & Inference Backend Naming Convention**: Standardized every entry in Settings (**Detection Model** and **Inference Backend**), the Settings header status, and the header telemetry strip around `<Model> — <Tier> [<CPU/GPU> · <Backend>]`. Changing **Inference Backend** or probing host hardware dynamically updates every model option's `[CPU/GPU · Backend]` badge live in Settings without auto-submitting.

## 1.0.29
- **Per-Line Modes (`Count + Speed` / `Count Only` / `Speed Only`)**: Added a dedicated `line_role` (`"both"`, `"count"`, `"speed"`) to traffic lines with automatic SQLite schema migration and a `+ Draw Speed Line` action plus 1-click mode cycling pill in **Zones & Lines**.
- **Clean Counting vs. Speed-Trap Separation**: `Count Only` lines track `A → B` / `B → A` crossings without pairing into 2-line speed timing; `Speed Only` lines measure vehicle speed in mph between speed lines without incrementing cross counters, cluttering KPI totals, or appearing in daily flow reports, and render as dashed amber `SPEED:` overlays on live/snapshot views.

## 1.0.28
- **Intel QuickSync VA-API Hardware Decode & Demand-Driven Frame Skipping**: Added `_FFmpegVAAPICapture` (`ffmpeg -hwaccel vaapi -hwaccel_device /dev/dri/renderD*` with `intel-media-va-driver-non-free`, `ffmpeg`, and `libva-drm2` cached cleanly after the `.venv` Docker layer) plus demand-driven `cap.grab()` / `cap.retrieve()` skipping so unneeded YUV→BGR decodes are bypassed when idle or between inference intervals.
- **OpenVINO Native FP16 & Persistent GPU Shader Cache**: GPU OpenVINO exports (`yolo` and `D-FINE`) now compile in native Intel Xe-LP **FP16** (`half=True` / `INFERENCE_PRECISION_HINT=f16`) with persistent OpenCL (`NEO_CACHE_DIR=/data/models/neo_cache`) and OpenVINO (`CACHE_DIR=/data/models/ov_cache`) caches for faster startup and lower GPU heat.
- **YOLO26s Default, Auto iGPU Selection & Lower Preview Load**: Updated default detection model to `yolo26s.pt` (NMS-free high accuracy), made `inference_backend="auto"` automatically pick `openvino_gpu` when an Intel iGPU is present, throttled browser preview JPEG rendering during `Gate: Idle` (~2.5 FPS with immediate motion wakeup), and respected `960px` preview width.
- **Thread-Capped ALPR (`RapidOCR`)**: Capped RapidOCR CPU threads (`intra_op_num_threads<=2`, `inter_op_num_threads=1`, `use_cls=False`), trimmed the top 15% roof/sky region, and downscaled oversized >960px 4K vehicle crops to prevent CPU spikes during violation snapshots.

## 1.0.27
- **Fixed-Position Header Telemetry Strip**: Replaced the free-flowing `#streamInfo` header string with fixed-slot flex spans (`.telemetry-strip` / `.t-slot`) and `font-variant-numeric: tabular-nums` + fixed decimal formatting (`.toFixed(2)` FPS, `.toFixed(1)` CPU/RAM/temp/mph) and separated Motion Gate (`Gate: Idle` / `Gate: Active`) into its own reserved slot so values stay aligned instead of shifting left and right on updates.

## 1.0.26
- **Fix Add-on Docker Build Copy Step**: Removed gitignored `yolov8n.pt` from `car_tracker/Dockerfile` and root `Dockerfile` (YOLO weights are downloaded into `/data/models/` at runtime), while keeping the cached `ubuntu:24.04` + `kobuk-team/intel-graphics` + `.venv` dependency layers intact for fast rebuilds.

## 1.0.25
- **Cached Python Dependency Docker Layer (Fast Add-on Updates)**: Moved `pyproject.toml` and `uv.lock` into the add-on Docker build context ahead of `COPY config.yaml` and switched startup (`car_tracker/run.sh`) to `uv run --no-sync` so future version bumps reuse the cached `.venv` layer (PyTorch, OpenVINO, ONNX Runtime, OpenCV) in seconds instead of re-downloading ~1.5 GB of Python packages on every update.
- **Intel N150 (`0x8086:0x46d4`) Twin Lake OpenCL Driver via Canonical Intel Graphics PPA**: Added `ppa:kobuk-team/intel-graphics` (`intel-opencl-icd 26.40+` + `libze-intel-gpu1`) on Ubuntu 24.04 `amd64` because stock `universe` `intel-opencl-icd 23.43` predates Twin Lake `0x46d4`, and fixed `/proc/cpuinfo` `model name` parsing so x86 `model: 190` no longer shadows `Intel(R) N150`.

## 1.0.24
- **Fix Intel iGPU OpenCL / Level Zero Container Drivers**: Switched the container base image from `python:3.12-slim` (Debian 13 Trixie, which removed `intel-opencl-icd` and caused silent OpenCL install failure) to `ubuntu:24.04` (Noble LTS) with `ocl-icd-libopencl1`, `intel-opencl-icd`, `libze-intel-gpu1`, `libze1`, and `clinfo` on `amd64` so Intel N95/N100/N150/N305/Core iGPUs are properly exposed to OpenVINO.
- **Hardware & GPU Diagnostics (`GET /api/system/hardware`)**: Inspects CPU model/cores, virtualization flags, PCI GPU IDs, `/dev/dri/*` render-node permissions, OpenCL ICDs, and OpenVINO/ONNX/PyTorch backends; surfaces exact GPU fallback reasons in `#streamInfo` and Settings (`Check GPU`), and prioritizes `x86_pkg_temp`/`coretemp`/`k10temp` over fixed `acpitz` (`27.8°C`) thermal sensors.

## 1.0.23
- **Scheduled After-Hours Person Watch**: Added configurable time-window person monitoring (`person_zone_enabled`, `person_zone_start_time`, `person_zone_end_time`, `person_zone_dwell_seconds`, `person_zone_scope`) supporting both same-day and cross-midnight schedules (default `00:00`–`05:00`), drawable `person_zone` polygons (`+ Draw Person Zone` in purple), and four watch scopes (`person_zones_or_frame`, `person_zones`, `red_zones`, `full_frame`).
- **Person Alerts Counter, Snapshots & Home Assistant Entity**: Added persistent `person_alerts` counter in `stats_state`, 7th KPI card (`Person Zone Alerts`) with active/scheduled badge, high-resolution annotated `person_*.jpg` snapshots (skipping vehicle ALPR OCR) with live dwell updates, webhook alerts, daily summary totals (`person_alerts_today`), `car_tracker_person_alert` HA events, and `sensor.car_tracker_person_alerts`.
- **Symmetric 3 × 2 Desktop Bento Settings**: Added the Scheduled Person Watch card (with `?` tooltips on all 5 new settings) and aligned all 6 Settings cards into an even `3 × 2` desktop / `2 × 3` tablet grid with `?v=21` static asset cache-busting.

## 1.0.22
- **Custom Home Assistant Add-on Icon, Detail Logo & Web Favicon**: Added Retina-crisp transparent squircle icon (`car_tracker/icon.png`, `256x256`) and detail-page banner (`car_tracker/logo.png`, `500x200`) for Home Assistant's Add-ons store and installed-apps grid, plus matching browser favicon (`static/icon.png`).

## 1.0.21
- **Zero-Scroll Desktop Bento Settings**: Widened the Settings dialog (`min(1520px, 100dvw - 32px)`) into a 6-track bento layout (`3 + 2` cards), moved the active runtime status summary into the modal header, placed action buttons inline in card headers, streamlined toggle rows into single-line pills (relying on `?` tooltips instead of verbose subtext), and aligned all 4 emergency vehicle thresholds in one compact row so the entire Settings menu fits on desktop without vertical scrolling.

## 1.0.20
- **Escape Dismissal & Native Dialogs**: Migrated the Settings, Violation Snapshot, and Bird's-Eye View popups to native `<dialog closedby="any">` modals so any popup (dialogs, Zones dropdown, active tooltips, and active canvas drawing) can be dismissed with the `Escape` key or backdrop click, while ensuring `Escape` dismisses an open tooltip before closing Settings.
- **Wide Multi-Column Settings Layout**: Expanded the Settings modal (`min(1140px, 100dvw - 32px)`), pinned the header and footer (`Cancel` / `Save & Apply`) so actions stay visible without scrolling, and organized all 32 settings into 5 grouped cards across a 2-column desktop layout that collapses cleanly on smaller screens.
- **Per-Setting Tooltips**: Added accessible `?` toggletips (`popover="hint"` + `interestfor` + hover/focus/tap fallbacks) on every setting explaining what it controls, valid ranges, and recommended values.

## 1.0.19
- **YOLO26 & D-FINE Model Options**: Added `yolo26n.pt`, `yolo26s.pt` (end-to-end NMS-free YOLO via Ultralytics 8.4), `dfine_n.onnx`, and `dfine_s.onnx` (HuggingFace `onnx-community` D-FINE transformers with standalone ByteTrack + `TrackStabilizer`) in Settings and `ObjectDetector`, supporting live hot-reloading and ONNX Runtime / OpenVINO execution.
- **Stream-Matched Rectangular Input Sizing**: Automatically computes `(in_h, in_w)` matching the camera's stream aspect ratio without upscaling low-resolution sub-streams (e.g. `640x360` sub-stream runs at `640x384` instead of `960x960`), caches rectangular ONNX/OpenVINO exports by `(in_h, in_w)`, and surfaces the active `WxH` inference geometry in the UI.
- **Smart Motion-Gated Inference**: Adds a lightweight OpenCV `MotionGate` (excluding configured Exclusion Masks) that runs neural-network inference at full `target_fps` on motion and throttles static scenes down to a `0.5 FPS` (`2.0s`) verification heartbeat while keeping active Red-Zone dwell timers and snapshots ticking continuously.

## 1.0.18
- **Compact Header Stream Label**: Replaced the full masked stream URL in `#streamInfo` with a concise protocol and stream-quality label (`RTMP • Sub Stream`, `RTSP • Main Stream`, `HTTP Snap`, or `File (<filename>)`) backed by `StreamWorker.get_source_summary()` and structured `stream_protocol` / `stream_quality` / `source_label` fields in `GET /api/stats` (while keeping `source` for API compatibility).

## 1.0.17
- **Live YOLO Model, Inference Backend & Image Size Hot-Reload**: Changing `yolo_model`, `inference_backend`, or static `yolo_imgsz` (`onnx`/`openvino`/`openvino_gpu`) in Settings now reloads the `ObjectDetector` dynamically inside the live processing loop without restarting the container or dropping active RTSP / HTTP Snap connections.
- **Explicit Backend Honors on GPU Hosts**: Fixed `ObjectDetector` initialization so explicit `onnx`, `openvino`, `openvino_gpu`, and `pytorch` selections are honored even on hosts with Apple MPS or CUDA GPUs, recorded with fallback reasons when a requested runtime is unavailable, and enforced static `imgsz` on ONNX/OpenVINO inference calls.
- **Track & Stream Resilience + UI Runtime Visibility**: Preserved canonical `TrackStabilizer` vehicle IDs across detector swaps (`clear_raw_bindings`), kept the worker alive if initial model load fails until updated in Settings, guarded against tight reload retry loops on failed models, and surfaced active model/backend/`imgsz`/loading/fallback status in `#streamInfo` and Settings (`#cvRuntimeHint`).

## 1.0.16
- **UI Header Version Badge**: Displays the running add-on version (e.g. `v1.0.16`) as a pill badge beside the header title in the web UI, backed by a dynamic `version` field in `GET /api/stats` and FastAPI metadata sourced from `config.yaml` (and `/tmp/addon_config.yaml` inside the Home Assistant add-on container), keeping the Web UI, API, and Home Assistant `"Car Tracker"` device software version automatically in sync.

## 1.0.15
- **Continuous Red-Zone Parked Timer**: Keeps counting parked duration continuously after a vehicle passes the dwell threshold (`30s` → `45s` → `2m 15s`) for the entire time it remains parked in a red zone—across the live video overlay (`ILLEGAL PARK`), Home Assistant's `max_dwell_seconds` attribute, and the Violations log (`Parked:` duration)—with seamless track-ID handover and clean finalization when the car departs.
- **Live Start-of-Violation Snapshot Updates**: Retains a lightweight compressed JPEG of the clean start-of-violation high-res frame in memory (~300 KB per active violator, instead of a heavy MP4 video ring buffer) and atomically re-renders the initial violation snapshot every 5 seconds and upon vehicle departure so the captured start-of-violation photo always shows the updated total parked duration.

## 1.0.14
- **Single "Car Tracker" Home Assistant Device**: Registers a native `"Car Tracker"` device in Home Assistant's Device Registry and attaches all 8 entities (`sensor.car_tracker_vehicles_today`, `sensor.car_tracker_people_today`, `sensor.car_tracker_active_vehicles`, `sensor.car_tracker_last_speed`, `sensor.car_tracker_avg_speed`, `sensor.car_tracker_violations_total`, `sensor.car_tracker_last_plate`, and `binary_sensor.car_tracker_red_zone_occupied`) under that single device with persistent `unique_id`s, automatic cleanup of legacy standalone `/api/states` entities, batch webhook updates, and automatic re-registration if the device is ever deleted.

## 1.0.13
- **Zones & Lines Dropdown Menu UI**: Consolidated the 7 Live Camera Feed action buttons (`+ Draw Red Zone`, `Auto-Detect Red Curb`, `+ Exclusion Mask`, `+ Draw Traffic Line`, `3D Ground Grid`, `Grid: ON/OFF`, `Bird's-Eye View`) and both `Configured Red Zones` and `Configured Traffic Flow Lines` lists into a single anchored `Zones & Lines` dropdown popover (responsive bottom sheet on mobile), keeping the active drawing buttons (`Undo Point`, `Rename`, `Finish & Save`, `Cancel`) visible inline only while drawing or editing.
- **Static Car Tracking & Count Reliability**: Fixed YOLO+ByteTrack two-stage low-confidence tracking (`fuse_score: False`, `track_buffer: 150`, `agnostic_nms=True`) and added a post-tracker `TrackStabilizer` with spatial ID stitching, confidence-weighted vehicle class voting, adaptive bounding-box smoothing, and short dropout coasting so static parked cars stop flickering IDs/labels or inflating `Unique Vehicles`.
- **Dwell Grace & Anti-Jitter Line Crossings**: Added a 5s dropout grace period and stationary spatial deduplication in `RedZoneMonitor`, minimum-travel + perpendicular-deadband hysteresis in `LineCrossingMonitor` so parked cars jittering over a counting line no longer trigger false crossings, and persistent `/data/models/` caching for downloaded YOLO `.pt` weights.

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
