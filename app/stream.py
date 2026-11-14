from __future__ import annotations

import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import RuntimeConfig, Settings, build_rtsp_url, mask_rtsp_url
from app.database import Database
from app.detector import (
    EmergencyVehicleClassifier,
    HotspotTracker,
    LineCrossingMonitor,
    ObjectDetector,
    RedZoneMonitor,
    annotate_frame,
)

# Force RTSP over TCP to prevent H.264/H.265 packet-loss smearing over WiFi
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


def resize_for_preview(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale frame proportionally if wider than max_width."""
    h, w = frame.shape[:2]
    if w <= max_width or max_width <= 0:
        return frame
    scale = max_width / float(w)
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_AREA)


def test_stream_connection(source: str, timeout_sec: float = 25.0) -> dict[str, Any]:
    """Open a stream or video file briefly to verify connectivity and resolution."""
    cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        return {"ok": False, "error": f"Could not open stream: {mask_rtsp_url(source)}"}

    deadline = time.monotonic() + timeout_sec
    ok = False
    frame = None
    while time.monotonic() < deadline:
        ok, frame = cap.read()
        if ok and frame is not None:
            break
        time.sleep(0.1)
    cap.release()

    if not ok or frame is None:
        return {"ok": False, "error": f"Connected, but no frames received from {mask_rtsp_url(source)}"}

    h, w = frame.shape[:2]
    return {
        "ok": True,
        "width": int(w),
        "height": int(h),
        "source": mask_rtsp_url(source),
    }


class StreamWorker:
    """Background worker that reads the NVR/video stream, runs YOLO+ByteTrack, and updates stats."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self.settings = settings
        self.db = db
        self.snapshots_dir = settings.snapshots_path
        self.snapshots_dir.mkdir(parents=True, exist_ok=True)

        saved_cfg = db.get_saved_config()
        if saved_cfg:
            self.config = RuntimeConfig(**saved_cfg)
        else:
            self.config = RuntimeConfig.from_settings(settings)
            db.save_config(self.config.model_dump())

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._reconnect_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._detector: ObjectDetector | None = None
        self.monitor = RedZoneMonitor()
        self.line_monitor = LineCrossingMonitor()
        self.hotspot = HotspotTracker()
        self.emergency_classifier = EmergencyVehicleClassifier()

        self._latest_jpeg: bytes | None = None
        self._stream_status: str = "initializing"
        self._last_error: str = ""
        self._resolution: tuple[int, int] = (0, 0)
        self._measured_fps: float = 0.0
        self._active_cars: int = 0
        self._active_people: int = 0
        self._active_dwells: list[dict[str, Any]] = []
        self._last_violation: dict[str, Any] | None = None

    def get_source_url(self, cfg: RuntimeConfig | None = None) -> str:
        c = cfg or self.config
        if c.test_video_path.strip():
            return c.test_video_path.strip()
        return build_rtsp_url(
            host=c.nvr_host,
            port=c.nvr_port,
            channel=c.nvr_channel,
            stream_type=c.nvr_stream_type,
            user=c.nvr_user,
            password=c.nvr_pass,
        )

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="stream-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def update_config(self, new_cfg: RuntimeConfig) -> None:
        old_source = self.get_source_url(self.config)
        new_source = self.get_source_url(new_cfg)
        with self._lock:
            self.config = new_cfg
            self.db.save_config(new_cfg.model_dump())
        self.emergency_classifier.reset()
        if old_source != new_source:
            self._reconnect_event.set()

    def toggle_heatmap(self, enabled: bool | None = None) -> bool:
        with self._lock:
            new_val = (not self.config.heatmap_enabled) if enabled is None else enabled
            self.config.heatmap_enabled = new_val
            self.db.save_config(self.config.model_dump())
            return new_val

    def reset_heatmap(self) -> None:
        with self._lock:
            self.hotspot.reset()

    def get_latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._latest_jpeg

    def get_live_state(self) -> dict[str, Any]:
        stats = self.db.get_stats_state()
        with self._lock:
            return {
                "stream_status": self._stream_status,
                "last_error": self._last_error,
                "source": mask_rtsp_url(self.get_source_url()),
                "resolution": {"width": self._resolution[0], "height": self._resolution[1]},
                "fps": round(self._measured_fps, 2),
                "active_cars": self._active_cars,
                "active_people": self._active_people,
                "unique_cars": stats["unique_cars"],
                "total_cars": stats["total_cars"],
                "unique_people": stats["unique_people"],
                "total_people": stats["total_people"],
                "violations_total": self.db.count_violations(),
                "counting_lines": [ln.model_dump() for ln in self.db.get_counting_lines()],
                "active_dwells": list(self._active_dwells),
                "heatmap_enabled": self.config.heatmap_enabled,
                "last_violation": self._last_violation,
            }

    def _ensure_detector(self) -> ObjectDetector:
        if self._detector is None:
            self._set_status("loading_model", "")
            self._detector = ObjectDetector(self.settings.yolo_model)
        return self._detector

    def _set_status(self, status: str, error: str = "") -> None:
        with self._lock:
            self._stream_status = status
            self._last_error = error

    def _render_placeholder(self, message: str) -> bytes:
        canvas = np.zeros((540, 960, 3), dtype=np.uint8)
        canvas[:] = (24, 24, 32)
        cv2.putText(canvas, "Car Tracker - Stream Standby", (40, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (220, 220, 235), 2)
        cv2.putText(canvas, message[:90], (40, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 160, 180), 1)
        ok, buf = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return buf.tobytes() if ok else b""

    def _run_loop(self) -> None:
        with self._lock:
            self._latest_jpeg = self._render_placeholder("Loading YOLO model...")

        try:
            detector = self._ensure_detector()
        except Exception as exc:
            self._set_status("error", f"Failed to load YOLO model: {exc}")
            with self._lock:
                self._latest_jpeg = self._render_placeholder(f"Model error: {exc}")
            return

        backoff = 1.0
        while not self._stop_event.is_set():
            self._reconnect_event.clear()
            cfg = self.config
            source = self.get_source_url(cfg)
            is_file = bool(cfg.test_video_path.strip())

            if not is_file and not cfg.nvr_pass and not cfg.nvr_user:
                self._set_status("waiting_for_credentials", "Enter NVR username/password in Settings.")
                with self._lock:
                    self._latest_jpeg = self._render_placeholder("Configure NVR credentials or Test Video in Settings")
                self._reconnect_event.wait(timeout=2.0)
                continue

            self._set_status("connecting", "")
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                self._set_status("reconnecting", f"Cannot open {mask_rtsp_url(source)}")
                with self._lock:
                    self._latest_jpeg = self._render_placeholder(f"Reconnecting to {mask_rtsp_url(source)}...")
                self._reconnect_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, 15.0)
                continue

            backoff = 1.0
            self._set_status("connected", "")
            last_infer_ts = 0.0

            while not self._stop_event.is_set() and not self._reconnect_event.is_set():
                cfg = self.config
                interval = 1.0 / max(cfg.target_fps, 0.5)

                if is_file:
                    ok, frame = cap.read()
                    if not ok:
                        # Loop test video seamlessly
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ok, frame = cap.read()
                        if not ok:
                            break
                    now = time.monotonic()
                    sleep_time = interval - (now - last_infer_ts)
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 0.25))
                        continue
                else:
                    # Drain RTSP buffer via grab() so we always process the newest frame
                    grabbed = cap.grab()
                    if not grabbed:
                        self._set_status("reconnecting", "Stream interrupted, reconnecting...")
                        break
                    now = time.monotonic()
                    if (now - last_infer_ts) < interval:
                        continue
                    ok, frame = cap.retrieve()
                    if not ok or frame is None:
                        continue

                t0 = time.monotonic()
                dt = t0 - last_infer_ts if last_infer_ts > 0 else interval
                last_infer_ts = t0

                try:
                    h, w = frame.shape[:2]
                    zones = self.db.get_zones()
                    counting_lines = self.db.get_counting_lines()
                    tracks = detector.detect_and_track(
                        frame, imgsz=cfg.yolo_imgsz, conf=cfg.confidence_threshold
                    )
                    if cfg.exempt_emergency_vehicles:
                        self.emergency_classifier.update_detections(
                            frame,
                            tracks,
                            min_saturation=cfg.emergency_min_saturation,
                            min_value=cfg.emergency_min_value,
                            strobe_swing=cfg.emergency_strobe_swing,
                            fire_red_ratio=cfg.emergency_fire_red_ratio,
                        )
                    self.hotspot.update(tracks, w, h)

                    now_ts = time.time()
                    step = self.monitor.process_tracks(
                        tracks=tracks,
                        frame_w=w,
                        frame_h=h,
                        zones=zones,
                        now_ts=now_ts,
                        dwell_seconds=cfg.dwell_seconds,
                        movement_threshold=cfg.movement_threshold,
                        min_track_frames=cfg.min_track_frames,
                        exempt_emergency_vehicles=cfg.exempt_emergency_vehicles,
                    )
                    line_incs = self.line_monitor.process_tracks(
                        tracks=tracks,
                        frame_w=w,
                        frame_h=h,
                        lines=counting_lines,
                        now_ts=now_ts,
                    )
                    if line_incs:
                        self.db.increment_line_counts(line_incs)
                        counting_lines = self.db.get_counting_lines()

                    if (
                        step.new_unique_cars
                        or step.new_total_cars
                        or step.new_unique_people
                        or step.new_total_people
                    ):
                        self.db.increment_stats(
                            unique_cars=step.new_unique_cars,
                            total_cars=step.new_total_cars,
                            unique_people=step.new_unique_people,
                            total_people=step.new_total_people,
                        )

                    annotated_full = annotate_frame(
                        frame=frame,
                        tracks=tracks,
                        track_statuses=step.track_statuses,
                        zones=zones,
                        dwell_threshold=cfg.dwell_seconds,
                        hotspot=self.hotspot,
                        show_heatmap=cfg.heatmap_enabled,
                        counting_lines=counting_lines,
                        annotation_scale=cfg.annotation_scale,
                    )

                    # Save full-resolution snapshot for any new red-zone violations
                    for v in step.new_violations:
                        ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                        fname = f"violation_{ts_slug}_t{v.track_id}.jpg"
                        snap_path = self.snapshots_dir / fname
                        cv2.imwrite(str(snap_path), annotated_full, [cv2.IMWRITE_JPEG_QUALITY, 92])
                        rec = self.db.add_violation(
                            track_id=v.track_id,
                            zone_id=v.zone_id,
                            zone_name=v.zone_name,
                            vehicle_class=v.vehicle_class,
                            started_at=v.started_at,
                            flagged_at=v.flagged_at,
                            dwell_seconds=v.dwell_seconds,
                            snapshot_filename=fname,
                        )
                        with self._lock:
                            self._last_violation = rec.model_dump()

                    preview = resize_for_preview(annotated_full, cfg.preview_max_width)
                    ok_jpg, jpg_buf = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 80])

                    dwells = [
                        {"track_id": tid, **st}
                        for tid, st in step.track_statuses.items()
                        if st.get("in_red_zone")
                    ]

                    with self._lock:
                        if ok_jpg:
                            self._latest_jpeg = jpg_buf.tobytes()
                        self._resolution = (int(w), int(h))
                        self._measured_fps = 1.0 / max(dt, 1e-3)
                        self._active_cars = step.active_cars
                        self._active_people = step.active_people
                        self._active_dwells = dwells
                except Exception as exc:
                    self._set_status("error", f"Inference error: {exc}")
                    time.sleep(1.0)

            cap.release()
