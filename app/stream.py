from __future__ import annotations

import copy
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover
    psutil = None

from app.alpr import LicensePlateReader
from app.config import RuntimeConfig, Settings, build_rtmp_url, build_rtsp_url, mask_rtsp_url
from app.database import Database
from app.detector import (
    EmergencyVehicleClassifier,
    HotspotTracker,
    LineCrossingMonitor,
    ObjectDetector,
    RedZoneMonitor,
    TrackedDetection,
    annotate_frame,
    filter_excluded_tracks,
)
from app.ha_integration import HomeAssistantPublisher

# Force RTSP over TCP to prevent H.264/H.265 packet-loss smearing over WiFi
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")


class ReolinkHTTPSnapClient:
    """Fetches JPEG frames from a Reolink NVR via HTTP/HTTPS Login + Snap API."""

    def __init__(self, host: str, channel: int, user: str, password: str) -> None:
        import requests

        self.host = host
        self.channel_idx = max(0, channel - 1)
        self.user = user
        self.password = password
        self.token: str | None = None
        self.scheme: str = "http"
        self._session = requests.Session()

    def _login(self) -> bool:
        payload = [
            {
                "cmd": "Login",
                "action": 0,
                "param": {"User": {"userName": self.user, "password": self.password}},
            }
        ]
        for scheme in ("http", "https"):
            try:
                r = self._session.post(
                    f"{scheme}://{self.host}/cgi-bin/api.cgi?cmd=Login",
                    json=payload,
                    timeout=5,
                    verify=False,
                )
                data = r.json()
                if isinstance(data, list) and data and data[0].get("code") == 0:
                    self.token = data[0]["value"]["Token"]["name"]
                    self.scheme = scheme
                    return True
            except Exception:
                continue
        return False

    def fetch_frame(self) -> np.ndarray | None:
        if not self.token and not self._login():
            return None

        for attempt in range(2):
            try:
                r = self._session.get(
                    f"{self.scheme}://{self.host}/cgi-bin/api.cgi",
                    params={
                        "cmd": "Snap",
                        "channel": self.channel_idx,
                        "rs": str(int(time.time() * 1000)),
                        "token": self.token,
                    },
                    timeout=6,
                    verify=False,
                )
                if "image/" in r.headers.get("content-type", "").lower() or r.content[:2] == b"\xff\xd8":
                    arr = np.frombuffer(r.content, np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        return frame
                # Token may have expired; re-login once
                if attempt == 0 and self._login():
                    continue
                return None
            except Exception:
                return None
        return None


def resize_for_preview(frame: np.ndarray, max_width: int) -> np.ndarray:
    """Downscale frame proportionally if wider than max_width."""
    h, w = frame.shape[:2]
    if w <= max_width or max_width <= 0:
        return frame
    scale = max_width / float(w)
    new_h = max(1, int(h * scale))
    return cv2.resize(frame, (max_width, new_h), interpolation=cv2.INTER_LINEAR)


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
        self._ha_wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._report_thread: threading.Thread | None = None
        self._ha_thread: threading.Thread | None = None

        self._detector: ObjectDetector | None = None
        self.monitor = RedZoneMonitor()
        self.line_monitor = LineCrossingMonitor()
        self.hotspot = HotspotTracker()
        self.emergency_classifier = EmergencyVehicleClassifier()
        self.alpr = LicensePlateReader()
        self.ha_publisher = HomeAssistantPublisher(db=db)

        self._latest_jpeg: bytes | None = None
        self._latest_raw_frame: np.ndarray | None = None
        self._stream_status: str = "initializing"
        self._last_error: str = ""
        self._resolution: tuple[int, int] = (0, 0)
        self._measured_fps: float = 0.0
        self._active_cars: int = 0
        self._active_people: int = 0
        self._active_dwells: list[dict[str, Any]] = []
        self._last_violation: dict[str, Any] | None = None
        self._last_plate: dict[str, Any] = {"plate": "", "confidence": 0.0, "updated_at": ""}
        self._prev_red_zone_occupied: bool = False
        self._latest_overlay: tuple[
            list[TrackedDetection], dict[int, dict[str, Any]], list[Any], list[Any]
        ] = ([], {}, [], [])
        self._last_preview_ts: float = 0.0
        self._last_viewer_ts: float = 0.0
        self._snap_client: ReolinkHTTPSnapClient | None = None
        self._active_violation_snaps: dict[tuple[int, str], dict[str, Any]] = {}

    def get_source_url(self, cfg: RuntimeConfig | None = None) -> str:
        c = cfg or self.config
        if c.test_video_path.strip():
            return c.test_video_path.strip()
        if c.stream_mode == "rtmp":
            return build_rtmp_url(
                host=c.nvr_host,
                channel=c.nvr_channel,
                stream_type=c.nvr_stream_type,
                user=c.nvr_user,
                password=c.nvr_pass,
            )
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
        self._ha_wake_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="stream-worker", daemon=True)
        self._thread.start()
        self._report_thread = threading.Thread(
            target=self._daily_report_loop, name="daily-report", daemon=True
        )
        self._report_thread.start()
        self._ha_thread = threading.Thread(
            target=self._ha_sync_loop, name="ha-sync", daemon=True
        )
        self._ha_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._ha_wake_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._report_thread:
            self._report_thread.join(timeout=2.0)
        if self._ha_thread:
            self._ha_thread.join(timeout=2.0)

    def update_config(self, new_cfg: RuntimeConfig) -> None:
        old_source = self.get_source_url(self.config)
        old_mode = self.config.stream_mode
        with self._lock:
            self.config = new_cfg
            self.db.save_config(new_cfg.model_dump())
        self.emergency_classifier.reset()
        self._snap_client = None
        self._ha_wake_event.set()
        new_source = self.get_source_url(new_cfg)
        if old_source != new_source or old_mode != new_cfg.stream_mode:
            if old_source != new_source and self._detector is not None:
                self._detector.reset()
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
        self._last_viewer_ts = time.monotonic()
        with self._lock:
            return self._latest_jpeg

    def get_latest_raw_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_raw_frame is None else self._latest_raw_frame.copy()

    @staticmethod
    def _get_system_metrics() -> dict[str, Any]:
        """Collect host CPU, memory, process RAM, and Raspberry Pi CPU temperature."""
        cpu_percent: float | None = None
        memory_percent: float | None = None
        app_ram_mb: float | None = None
        cpu_temp_c: float | None = None

        if psutil is not None:
            try:
                cpu_percent = round(float(psutil.cpu_percent(interval=None)), 1)
                memory_percent = round(float(psutil.virtual_memory().percent), 1)
                app_ram_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
            except Exception:
                pass

        thermal_path = Path("/sys/class/thermal/thermal_zone0/temp")
        if thermal_path.exists():
            try:
                raw_temp = thermal_path.read_text(encoding="utf-8").strip()
                cpu_temp_c = round(int(raw_temp) / 1000.0, 1)
            except Exception:
                pass

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "app_ram_mb": app_ram_mb,
            "cpu_temp_c": cpu_temp_c,
        }

    def build_ha_payload(self) -> dict[str, Any]:
        stats = self.db.get_stats_state()
        speed = self.line_monitor.get_speed_summary()
        violations_total = self.db.count_violations()
        with self._lock:
            dwells = list(self._active_dwells)
            last_plate = dict(self._last_plate)
            last_violation = self._last_violation
            active_cars = self._active_cars
            active_people = self._active_people
        return {
            "vehicles_today": stats["total_cars"],
            "people_today": stats["total_people"],
            "unique_cars": stats["unique_cars"],
            "unique_people": stats["unique_people"],
            "active_cars": active_cars,
            "active_people": active_people,
            "last_speed_mph": speed.get("last_speed_mph"),
            "avg_speed_mph": speed.get("avg_speed_mph"),
            "max_speed_mph": speed.get("max_speed_mph"),
            "samples_count": speed.get("samples_count", 0),
            "violations_total": violations_total,
            "last_violation": last_violation,
            "last_plate": last_plate.get("plate", ""),
            "last_plate_confidence": last_plate.get("confidence", 0.0),
            "last_plate_updated_at": last_plate.get("updated_at", ""),
            "red_zone_occupied": len(dwells) > 0,
            "dwelling_vehicles": len(dwells),
            "max_dwell_seconds": max((float(d.get("dwell_seconds", 0.0)) for d in dwells), default=0.0),
        }

    def sync_ha_now(
        self, ha_url: str | None = None, ha_token: str | None = None
    ) -> dict[str, Any]:
        cfg = self.config
        url = cfg.ha_url if ha_url is None else ha_url
        token = cfg.ha_token if ha_token is None else ha_token
        return self.ha_publisher.publish_states(self.build_ha_payload(), ha_url=url, ha_token=token)

    def _ha_sync_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                cfg = self.config
                if cfg.ha_enabled and self.ha_publisher.resolve_target(cfg.ha_url, cfg.ha_token) is not None:
                    self.sync_ha_now(cfg.ha_url, cfg.ha_token)
            except Exception:
                pass
            self._ha_wake_event.wait(timeout=10.0)
            self._ha_wake_event.clear()

    def get_live_state(self) -> dict[str, Any]:
        stats = self.db.get_stats_state()
        system_metrics = self._get_system_metrics()
        ha_status = self.ha_publisher.get_status()
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
                "last_plate": dict(self._last_plate),
                "ha": ha_status,
                "system": system_metrics,
                "speed": self.line_monitor.get_speed_summary(),
                "inference_backend": self._detector.backend if self._detector is not None else self.config.inference_backend,
                "yolo_model": self.config.yolo_model,
            }

    def _ensure_detector(self) -> ObjectDetector:
        cfg = self.config
        desired_key = (
            cfg.yolo_model,
            cfg.yolo_imgsz if cfg.inference_backend in ("auto", "onnx", "openvino", "openvino_gpu") else 0,
            cfg.inference_backend,
        )
        if self._detector is None or getattr(self, "_detector_key", None) != desired_key:
            self._set_status("loading_model", "")
            self._detector = ObjectDetector(
                model_name=cfg.yolo_model,
                imgsz=cfg.yolo_imgsz,
                inference_backend=cfg.inference_backend,
                models_dir=self.settings.data_path / "models",
            )
            self._detector_key = desired_key
        return self._detector

    @staticmethod
    def _scale_tracks(
        tracks: list[TrackedDetection], src_w: int, src_h: int, dst_w: int, dst_h: int
    ) -> list[TrackedDetection]:
        sx = dst_w / float(max(src_w, 1))
        sy = dst_h / float(max(src_h, 1))
        if abs(sx - 1.0) <= 1e-3 and abs(sy - 1.0) <= 1e-3:
            return tracks
        return [
            TrackedDetection(
                track_id=t.track_id,
                class_id=t.class_id,
                class_name=t.class_name,
                confidence=t.confidence,
                bbox_xyxy=(
                    t.bbox_xyxy[0] * sx,
                    t.bbox_xyxy[1] * sy,
                    t.bbox_xyxy[2] * sx,
                    t.bbox_xyxy[3] * sy,
                ),
                is_emergency=t.is_emergency,
            )
            for t in tracks
        ]

    def _get_highres_violation_frame(
        self, fallback_frame: np.ndarray, cfg: RuntimeConfig
    ) -> np.ndarray:
        """Fetch a 4K HTTP Snap frame for violation evidence when running on a lower-res stream."""
        if fallback_frame.shape[1] >= 1920 or cfg.test_video_path.strip() or not cfg.nvr_host:
            return fallback_frame
        try:
            if self._snap_client is None:
                self._snap_client = ReolinkHTTPSnapClient(
                    cfg.nvr_host, cfg.nvr_channel, cfg.nvr_user, cfg.nvr_pass
                )
            hi_res = self._snap_client.fetch_frame()
            if hi_res is not None and hi_res.shape[1] > fallback_frame.shape[1]:
                return hi_res
        except Exception:
            pass
        return fallback_frame

    @staticmethod
    def _write_jpeg_atomic(snap_path: Path, frame: np.ndarray, quality: int = 92) -> bool:
        """Encode and atomically replace a JPEG snapshot file so readers never observe partial bytes."""
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return False
        snap_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = snap_path.with_name(f".{snap_path.name}.tmp")
        with open(tmp_path, "wb") as f:
            f.write(buf.tobytes())
        os.replace(tmp_path, snap_path)
        return True

    def _refresh_violation_snapshot(self, entry: dict[str, Any], dwell_seconds: float) -> bool:
        """Re-render the initial violation snapshot using the updated parked dwell duration."""
        clean_bytes: bytes = entry.get("clean_jpeg", b"")
        if not clean_bytes:
            return False
        base_img = cv2.imdecode(np.frombuffer(clean_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if base_img is None:
            return False

        orig_tid = int(entry["orig_track_id"])
        statuses = {k: dict(val) for k, val in entry["track_statuses"].items()}
        target_status = statuses.setdefault(
            orig_tid,
            {
                "in_red_zone": True,
                "dwell_seconds": round(float(dwell_seconds), 1),
                "violation": True,
                "zone_name": None,
                "emergency": False,
            },
        )
        target_status["in_red_zone"] = True
        target_status["violation"] = True
        target_status["dwell_seconds"] = round(float(dwell_seconds), 1)

        updated_img = annotate_frame(
            frame=base_img,
            tracks=entry["tracks"],
            track_statuses=statuses,
            zones=entry["zones"],
            dwell_threshold=float(entry["dwell_threshold"]),
            hotspot=None,
            show_heatmap=False,
            counting_lines=entry["counting_lines"],
            annotation_scale=float(entry["annotation_scale"]),
        )
        return self._write_jpeg_atomic(entry["snap_path"], updated_img, quality=92)

    def _render_preview_jpeg(
        self,
        frame: np.ndarray,
        tracks: list[TrackedDetection],
        track_statuses: dict[int, dict[str, Any]],
        zones: list[Any],
        counting_lines: list[Any],
        cfg: RuntimeConfig,
    ) -> bytes | None:
        """Scale frame to target preview resolution first, then annotate for crisp boxes/labels."""
        h, w = frame.shape[:2]
        target_w = min(max(cfg.preview_max_width, 1280), 1920) if cfg.preview_max_width > 0 else w
        if w > cfg.preview_max_width > 0:
            preview = resize_for_preview(frame, cfg.preview_max_width)
        elif 0 < w < target_w:
            scale = target_w / float(w)
            preview = cv2.resize(
                frame, (target_w, max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR
            )
        else:
            preview = frame
        ph, pw = preview.shape[:2]
        scaled_tracks = self._scale_tracks(tracks, w, h, pw, ph)

        annotated_preview = annotate_frame(
            frame=preview,
            tracks=scaled_tracks,
            track_statuses=track_statuses,
            zones=zones,
            dwell_threshold=cfg.dwell_seconds,
            hotspot=self.hotspot,
            show_heatmap=cfg.heatmap_enabled,
            counting_lines=counting_lines,
            annotation_scale=cfg.annotation_scale,
        )
        ok_jpg, jpg_buf = cv2.imencode(".jpg", annotated_preview, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpg_buf.tobytes() if ok_jpg else None

    def _maybe_publish_live_preview(self, frame: np.ndarray) -> None:
        """Publish an intermediate camera frame overlaid with the latest tracked boxes."""
        now = time.monotonic()
        if (now - self._last_preview_ts) < 0.05:
            return
        with self._lock:
            tracks, statuses, zones, lines = self._latest_overlay
            cfg = self.config
        jpg = self._render_preview_jpeg(frame, tracks, statuses, zones, lines, cfg)
        if jpg is not None:
            with self._lock:
                self._latest_jpeg = jpg
                self._last_preview_ts = now

    def _process_single_frame(
        self,
        frame: np.ndarray,
        detector: ObjectDetector,
        cfg: RuntimeConfig,
        dt: float,
        publish_preview: bool = True,
    ) -> None:
        try:
            h, w = frame.shape[:2]
            zones = self.db.get_zones()
            counting_lines = self.db.get_counting_lines()
            now_ts = time.time()
            tracks = detector.detect_and_track(
                frame, imgsz=cfg.yolo_imgsz, conf=cfg.confidence_threshold, now_ts=now_ts
            )
            tracks = filter_excluded_tracks(tracks, w, h, zones)
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
            calib = self.db.get_ground_calibration()
            line_incs = self.line_monitor.process_tracks(
                tracks=tracks,
                frame_w=w,
                frame_h=h,
                lines=counting_lines,
                now_ts=now_ts,
                calib=calib,
                movement_threshold=cfg.movement_threshold,
            )
            track_speeds = self.line_monitor.get_track_speeds()
            for tid, mph in track_speeds.items():
                if tid in step.track_statuses:
                    step.track_statuses[tid]["speed_mph"] = mph
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

            new_speeds = self.line_monitor.consume_new_speeds()
            speeding_events = (
                [sp for sp in new_speeds if sp["speed_mph"] > cfg.speed_limit_mph]
                if cfg.speed_capture_enabled
                else []
            )

            if step.new_violations or speeding_events:
                snap_base = self._get_highres_violation_frame(frame, cfg)
                sh, sw = snap_base.shape[:2]
                snap_tracks = self._scale_tracks(tracks, w, h, sw, sh)
                track_bbox_map = {t.track_id: t.bbox_xyxy for t in snap_tracks}
                plates_by_tid: dict[int, tuple[str, float]] = {}

                if cfg.alpr_enabled:
                    event_tids = {v.track_id for v in step.new_violations} | {
                        int(sp["track_id"]) for sp in speeding_events
                    }
                    for tid in event_tids:
                        plate, plate_conf = self.alpr.read_plate(snap_base, track_bbox_map.get(tid))
                        if plate:
                            plates_by_tid[tid] = (plate, plate_conf)
                            step.track_statuses.setdefault(tid, {})["license_plate"] = plate
                            with self._lock:
                                self._last_plate = {
                                    "plate": plate,
                                    "confidence": plate_conf,
                                    "updated_at": datetime.now(timezone.utc).isoformat(),
                                }

                clean_jpeg = b""
                if step.new_violations:
                    clean_source = (
                        self.hotspot.render_overlay(snap_base, alpha=0.45)
                        if cfg.heatmap_enabled and self.hotspot is not None
                        else snap_base
                    )
                    ok_clean, clean_buf = cv2.imencode(
                        ".jpg", clean_source, [cv2.IMWRITE_JPEG_QUALITY, 94]
                    )
                    if ok_clean:
                        clean_jpeg = clean_buf.tobytes()

                annotated_full = annotate_frame(
                    frame=snap_base,
                    tracks=snap_tracks,
                    track_statuses=step.track_statuses,
                    zones=zones,
                    dwell_threshold=cfg.dwell_seconds,
                    hotspot=self.hotspot,
                    show_heatmap=cfg.heatmap_enabled,
                    counting_lines=counting_lines,
                    annotation_scale=cfg.annotation_scale,
                )
                for v in step.new_violations:
                    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    fname = f"violation_{ts_slug}_t{v.track_id}.jpg"
                    snap_path = self.snapshots_dir / fname
                    self._write_jpeg_atomic(snap_path, annotated_full, quality=92)
                    plate_text = plates_by_tid.get(v.track_id, ("", 0.0))[0]
                    rec = self.db.add_violation(
                        track_id=v.track_id,
                        zone_id=v.zone_id,
                        zone_name=v.zone_name,
                        vehicle_class=v.vehicle_class,
                        started_at=v.started_at,
                        flagged_at=v.flagged_at,
                        dwell_seconds=v.dwell_seconds,
                        snapshot_filename=fname,
                        license_plate=plate_text,
                    )
                    if clean_jpeg:
                        now_mono = time.monotonic()
                        snap_entry: dict[str, Any] = {
                            "violation_id": rec.id,
                            "orig_track_id": int(v.track_id),
                            "zone_id": v.zone_id,
                            "snap_path": snap_path,
                            "clean_jpeg": clean_jpeg,
                            "tracks": copy.deepcopy(snap_tracks),
                            "track_statuses": copy.deepcopy(step.track_statuses),
                            "zones": copy.deepcopy(zones),
                            "counting_lines": copy.deepcopy(counting_lines),
                            "dwell_threshold": float(cfg.dwell_seconds),
                            "annotation_scale": float(cfg.annotation_scale),
                            "created_ts": now_mono,
                            "last_update_ts": now_mono,
                            "last_rendered_dwell": round(float(v.dwell_seconds), 1),
                        }
                        with self._lock:
                            expired_keys = [
                                k
                                for k, val in self._active_violation_snaps.items()
                                if (now_mono - float(val["created_ts"])) > 14400.0
                            ]
                            for ek in expired_keys:
                                self._active_violation_snaps.pop(ek, None)
                            while len(self._active_violation_snaps) >= 16:
                                oldest_key = min(
                                    self._active_violation_snaps,
                                    key=lambda k: float(self._active_violation_snaps[k]["created_ts"]),
                                )
                                self._active_violation_snaps.pop(oldest_key, None)
                            self._active_violation_snaps[(int(v.track_id), v.zone_id)] = snap_entry
                    with self._lock:
                        self._last_violation = rec.model_dump()
                    if cfg.webhook_url.strip():
                        self._dispatch_webhook(cfg.webhook_url.strip(), rec.model_dump(), snap_path)
                    if cfg.ha_enabled:
                        threading.Thread(
                            target=self.ha_publisher.fire_event,
                            args=("car_tracker_violation", rec.model_dump(), cfg.ha_url, cfg.ha_token),
                            daemon=True,
                        ).start()

                for sp in speeding_events:
                    now_iso = datetime.now(timezone.utc).isoformat()
                    ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                    fname = f"speeding_{ts_slug}_t{sp['track_id']}.jpg"
                    snap_path = self.snapshots_dir / fname
                    self._write_jpeg_atomic(snap_path, annotated_full, quality=92)
                    plate_text = plates_by_tid.get(int(sp["track_id"]), ("", 0.0))[0]
                    rec = self.db.add_violation(
                        track_id=sp["track_id"],
                        zone_id="speeding",
                        zone_name=f"Speeding {sp['speed_mph']:.1f} mph (limit {cfg.speed_limit_mph:g} mph)",
                        vehicle_class=sp["vehicle_class"],
                        started_at=now_iso,
                        flagged_at=now_iso,
                        dwell_seconds=float(sp["speed_mph"]),
                        snapshot_filename=fname,
                        license_plate=plate_text,
                    )
                    with self._lock:
                        self._last_violation = rec.model_dump()
                    if cfg.webhook_url.strip():
                        self._dispatch_webhook(cfg.webhook_url.strip(), rec.model_dump(), snap_path)
                    if cfg.ha_enabled:
                        event_payload = {**rec.model_dump(), "speed_mph": float(sp["speed_mph"])}
                        threading.Thread(
                            target=self.ha_publisher.fire_event,
                            args=("car_tracker_speeding", event_payload, cfg.ha_url, cfg.ha_token),
                            daemon=True,
                        ).start()

                self._ha_wake_event.set()

            if step.dwell_updates:
                now_mono = time.monotonic()
                any_completed = False
                for upd in step.dwell_updates:
                    with self._lock:
                        snap_entry = self._active_violation_snaps.get(upd.violation_key)
                    if snap_entry is None:
                        continue
                    rounded_dwell = round(float(upd.dwell_seconds), 1)
                    should_refresh = upd.completed or (
                        (now_mono - float(snap_entry["last_update_ts"])) >= 5.0
                        and int(round(rounded_dwell)) != int(round(float(snap_entry["last_rendered_dwell"])))
                    )
                    if should_refresh:
                        self.db.update_violation_dwell(snap_entry["violation_id"], rounded_dwell)
                        self._refresh_violation_snapshot(snap_entry, rounded_dwell)
                        snap_entry["last_update_ts"] = now_mono
                        snap_entry["last_rendered_dwell"] = rounded_dwell
                        with self._lock:
                            if (
                                self._last_violation is not None
                                and self._last_violation.get("id") == snap_entry["violation_id"]
                            ):
                                self._last_violation["dwell_seconds"] = rounded_dwell
                    if upd.completed:
                        with self._lock:
                            self._active_violation_snaps.pop(upd.violation_key, None)
                        any_completed = True
                if any_completed:
                    self._ha_wake_event.set()

            jpg_bytes = (
                self._render_preview_jpeg(
                    frame=frame,
                    tracks=tracks,
                    track_statuses=step.track_statuses,
                    zones=zones,
                    counting_lines=counting_lines,
                    cfg=cfg,
                )
                if publish_preview
                else None
            )

            dwells = [
                {"track_id": tid, **st}
                for tid, st in step.track_statuses.items()
                if st.get("in_red_zone")
            ]

            with self._lock:
                self._latest_raw_frame = frame
                self._latest_overlay = (list(tracks), dict(step.track_statuses), list(zones), list(counting_lines))
                if jpg_bytes is not None:
                    self._latest_jpeg = jpg_bytes
                    self._last_preview_ts = time.monotonic()
                self._resolution = (int(w), int(h))
                self._measured_fps = 1.0 / max(dt, 1e-3)
                self._active_cars = step.active_cars
                self._active_people = step.active_people
                self._active_dwells = dwells

            red_zone_occupied = len(dwells) > 0
            if red_zone_occupied != self._prev_red_zone_occupied:
                self._prev_red_zone_occupied = red_zone_occupied
                self._ha_wake_event.set()
        except Exception as exc:
            self._set_status("error", f"Inference error: {exc}")
            time.sleep(1.0)

    def _run_http_snap_fallback(self, detector: ObjectDetector) -> bool:
        """Poll Reolink HTTP Snap API in a background reader thread. Returns True if frames were served."""
        cfg = self.config
        client = ReolinkHTTPSnapClient(cfg.nvr_host, cfg.nvr_channel, cfg.nvr_user, cfg.nvr_pass)
        first = client.fetch_frame()
        if first is None:
            return False

        self._set_status("connected_http", "Using HTTP Snap fallback")
        snap_lock = threading.Lock()
        snap_state: dict[str, Any] = {"frame": first, "seq": 1, "alive": True}
        reader_stop = threading.Event()

        def _http_reader() -> None:
            failures = 0
            while not reader_stop.is_set() and not self._stop_event.is_set() and not self._reconnect_event.is_set():
                frm = client.fetch_frame()
                if frm is None:
                    failures += 1
                    if failures >= 5:
                        snap_state["alive"] = False
                        break
                    time.sleep(0.3)
                    continue
                failures = 0
                with snap_lock:
                    snap_state["frame"] = frm
                    snap_state["seq"] += 1

        reader_thread = threading.Thread(target=_http_reader, name="http-snap-reader", daemon=True)
        reader_thread.start()

        last_infer_ts = 0.0
        last_seq = 0
        try:
            while not self._stop_event.is_set() and not self._reconnect_event.is_set():
                if not snap_state["alive"]:
                    return True
                cfg = self.config
                interval = 1.0 / max(cfg.target_fps, 0.5)
                now = time.monotonic()
                if (now - last_infer_ts) < interval:
                    time.sleep(0.01)
                    continue
                with snap_lock:
                    seq = snap_state["seq"]
                    frame = snap_state["frame"]
                if frame is None or seq == last_seq:
                    time.sleep(0.01)
                    continue
                last_seq = seq
                t0 = time.monotonic()
                dt = t0 - last_infer_ts if last_infer_ts > 0 else interval
                last_infer_ts = t0
                self._process_single_frame(frame, detector, cfg, dt)
        finally:
            reader_stop.set()

        return True

    def _set_status(self, status: str, error: str = "") -> None:
        with self._lock:
            self._stream_status = status
            self._last_error = error

    def _dispatch_webhook(self, url: str, record: dict[str, Any], snap_path: Path) -> None:
        def _send() -> None:
            try:
                import requests

                if record.get("zone_id") == "speeding":
                    msg = (
                        f"🚨 Speeding Alert: {record['vehicle_class'].upper()} #{record['track_id']} "
                        f"— {record['zone_name']}"
                    )
                else:
                    msg = (
                        f"🚨 Red-Zone Violation: {record['vehicle_class'].upper()} #{record['track_id']} "
                        f"in {record['zone_name']} (dwell: {record['dwell_seconds']:.0f}s)"
                    )
                if record.get("license_plate"):
                    msg = f"{msg} • 🪪 Plate: **{record['license_plate']}**"
                if "discord.com/api/webhooks" in url:
                    with open(snap_path, "rb") as f:
                        requests.post(
                            url,
                            data={"content": msg},
                            files={"file": (snap_path.name, f, "image/jpeg")},
                            timeout=8,
                        )
                else:
                    requests.post(url, json={"text": msg, "content": msg, "violation": record}, timeout=8)
            except Exception:
                pass

        threading.Thread(target=_send, daemon=True).start()

    @staticmethod
    def build_daily_report_message(summary: dict[str, Any]) -> str:
        lines_out = [
            f"📊 **Daily Car Tracker Report — {summary['date']}**",
            f"• **Vehicles Today:** {summary['vehicles_today']}",
            f"• **People Today:** {summary['people_today']}",
        ]
        peak = summary.get("peak_hour")
        if peak:
            hour_label = str(peak["hour_bucket"]).split("T")[-1]
            lines_out.append(f"• **Peak Hour:** {hour_label}:00 ({peak['vehicles_count']} vehicles)")
        lines_out.append(f"• **Red-Zone Violations Today:** {summary['violations_today']}")
        counting_lines = summary.get("counting_lines") or []
        if counting_lines:
            lines_out.append("• **Traffic Flow Lines (total A→B / B→A):**")
            for ln in counting_lines:
                a_b = int(ln.get("count_a_to_b", 0))
                b_a = int(ln.get("count_b_to_a", 0))
                lines_out.append(f"  - {ln['name']}: **{a_b + b_a}** ({a_b} / {b_a})")
        cum = summary.get("cumulative_stats") or {}
        lines_out.append(
            f"• **All-Time Totals:** {cum.get('unique_cars', 0)} unique vehicles, "
            f"{cum.get('unique_people', 0)} unique people"
        )
        return "\n".join(lines_out)

    def reset_all_counts(self) -> None:
        """Reset cumulative vehicle/people counts, hourly traffic, line crossings, and speed history."""
        self.db.reset_stats()
        self.db.reset_line_counts()
        self.monitor.reset()
        self.line_monitor.reset()
        with self._lock:
            self._active_violation_snaps.clear()
        self._ha_wake_event.set()

    def send_daily_report(
        self,
        webhook_url: str | None = None,
        date_str: str | None = None,
        reset_after: bool = True,
    ) -> dict[str, Any]:
        import requests

        url = (webhook_url if webhook_url is not None else self.config.webhook_url).strip()
        if not url:
            return {"ok": False, "error": "Webhook URL is empty."}

        summary = self.db.get_daily_summary(date_str=date_str)
        msg = self.build_daily_report_message(summary)
        payload = (
            {"content": msg}
            if "discord.com/api/webhooks" in url
            else {"text": msg, "content": msg, "daily_report": summary}
        )
        try:
            resp = requests.post(url, json=payload, timeout=10)
            if 200 <= resp.status_code < 300:
                if reset_after:
                    self.reset_all_counts()
                return {
                    "ok": True,
                    "status_code": resp.status_code,
                    "summary": summary,
                    "reset": reset_after,
                }
            return {"ok": False, "error": f"HTTP {resp.status_code}: {resp.text[:160]}"}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def _daily_report_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                cfg = self.config
                if cfg.daily_report_enabled and cfg.webhook_url.strip():
                    now_local = datetime.now()
                    today_str = now_local.strftime("%Y-%m-%d")
                    now_hhmm = now_local.strftime("%H:%M")
                    target_hhmm = (cfg.daily_report_time or "23:55").strip()
                    last_sent = self.db.get_meta("last_daily_report_date")
                    if now_hhmm >= target_hhmm and last_sent != today_str:
                        res = self.send_daily_report(webhook_url=cfg.webhook_url, date_str=today_str)
                        if res.get("ok"):
                            self.db.set_meta("last_daily_report_date", today_str)
            except Exception:
                pass
            self._stop_event.wait(timeout=30.0)

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

            if not is_file and cfg.stream_mode == "http_snap":
                if self._run_http_snap_fallback(detector):
                    backoff = 1.0
                    continue
                self._set_status("reconnecting", f"HTTP Snap failed for {cfg.nvr_host}")
                with self._lock:
                    self._latest_jpeg = self._render_placeholder(f"Reconnecting HTTP Snap to {cfg.nvr_host}...")
                self._reconnect_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, 15.0)
                continue

            self._set_status("connecting", "")
            with self._lock:
                self._latest_jpeg = self._render_placeholder(f"Connecting to {mask_rtsp_url(source)}...")
            cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                # Fall back to Reolink HTTP Snap API when RTSP cannot open
                if not is_file and cfg.stream_mode == "auto" and self._run_http_snap_fallback(detector):
                    backoff = 1.0
                    continue
                self._set_status("reconnecting", f"Cannot open {mask_rtsp_url(source)}")
                with self._lock:
                    self._latest_jpeg = self._render_placeholder(f"Reconnecting to {mask_rtsp_url(source)}...")
                self._reconnect_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, 15.0)
                continue

            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            backoff = 1.0
            last_infer_ts = 0.0
            first_frame_seen = False

            # Dedicated RTSP/RTMP reader thread drains cap.read() at full speed so FFmpeg never buffers.
            # A separate preview thread encodes only the newest frame strictly forward in time (no rubber-banding).
            rtsp_lock = threading.Lock()
            rtsp_state: dict[str, Any] = {"frame": None, "seq": 0, "alive": True}
            reader_stop = threading.Event()
            reader_thread: threading.Thread | None = None
            preview_thread: threading.Thread | None = None
            if not is_file:
                def _rtsp_reader() -> None:
                    try:
                        while (
                            not reader_stop.is_set()
                            and not self._stop_event.is_set()
                            and not self._reconnect_event.is_set()
                        ):
                            ok_r, frm = cap.read()
                            if not ok_r or frm is None:
                                rtsp_state["alive"] = False
                                break
                            with rtsp_lock:
                                rtsp_state["frame"] = frm
                                rtsp_state["seq"] += 1
                    finally:
                        cap.release()

                def _preview_publisher() -> None:
                    last_p_seq = 0
                    while (
                        not reader_stop.is_set()
                        and not self._stop_event.is_set()
                        and not self._reconnect_event.is_set()
                    ):
                        if (time.monotonic() - getattr(self, "_last_viewer_ts", 0.0)) > 5.0:
                            time.sleep(0.25)
                            continue
                        with rtsp_lock:
                            seq = rtsp_state["seq"]
                            frm = rtsp_state["frame"]
                        if frm is None or seq == last_p_seq:
                            time.sleep(0.02)
                            continue
                        last_p_seq = seq
                        with self._lock:
                            tracks, statuses, zones, lines = self._latest_overlay
                            pcfg = self.config
                        jpg = self._render_preview_jpeg(frm, tracks, statuses, zones, lines, pcfg)
                        if jpg is not None:
                            with self._lock:
                                self._latest_jpeg = jpg
                        time.sleep(0.08)

                reader_thread = threading.Thread(target=_rtsp_reader, name="rtsp-reader", daemon=True)
                reader_thread.start()
                preview_thread = threading.Thread(target=_preview_publisher, name="preview-pub", daemon=True)
                preview_thread.start()

            last_seq = 0
            while not self._stop_event.is_set() and not self._reconnect_event.is_set():
                cfg = self.config
                interval = 1.0 / max(cfg.target_fps, 0.5)

                if is_file:
                    ok, frame = cap.read()
                    if not ok:
                        # Loop test video seamlessly
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        detector.reset()
                        ok, frame = cap.read()
                        if not ok:
                            break
                    now = time.monotonic()
                    sleep_time = interval - (now - last_infer_ts)
                    if sleep_time > 0:
                        time.sleep(min(sleep_time, 0.25))
                        continue
                else:
                    if not rtsp_state["alive"]:
                        self._set_status("reconnecting", "Stream interrupted, reconnecting...")
                        break
                    now = time.monotonic()
                    if (now - last_infer_ts) < interval:
                        time.sleep(0.005)
                        continue
                    with rtsp_lock:
                        seq = rtsp_state["seq"]
                        frame = rtsp_state["frame"]
                    if frame is None or seq == last_seq:
                        time.sleep(0.01)
                        continue
                    last_seq = seq

                if not first_frame_seen:
                    first_frame_seen = True
                    self._set_status("connected", "")

                t0 = time.monotonic()
                dt = t0 - last_infer_ts if last_infer_ts > 0 else interval
                last_infer_ts = t0
                self._process_single_frame(frame, detector, cfg, dt, publish_preview=is_file)

            reader_stop.set()
            if preview_thread is not None:
                preview_thread.join(timeout=1.5)
            if reader_thread is not None:
                reader_thread.join(timeout=3.0)
            else:
                cap.release()
            if not first_frame_seen and not is_file and cfg.stream_mode == "auto" and not self._stop_event.is_set() and not self._reconnect_event.is_set():
                self._run_http_snap_fallback(detector)
