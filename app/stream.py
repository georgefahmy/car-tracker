from __future__ import annotations

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
        self._thread: threading.Thread | None = None
        self._report_thread: threading.Thread | None = None

        self._detector: ObjectDetector | None = None
        self.monitor = RedZoneMonitor()
        self.line_monitor = LineCrossingMonitor()
        self.hotspot = HotspotTracker()
        self.emergency_classifier = EmergencyVehicleClassifier()

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
        self._latest_overlay: tuple[
            list[TrackedDetection], dict[int, dict[str, Any]], list[Any], list[Any]
        ] = ([], {}, [], [])
        self._last_preview_ts: float = 0.0
        self._snap_client: ReolinkHTTPSnapClient | None = None

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
        self._thread = threading.Thread(target=self._run_loop, name="stream-worker", daemon=True)
        self._thread.start()
        self._report_thread = threading.Thread(
            target=self._daily_report_loop, name="daily-report", daemon=True
        )
        self._report_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._report_thread:
            self._report_thread.join(timeout=2.0)

    def update_config(self, new_cfg: RuntimeConfig) -> None:
        old_source = self.get_source_url(self.config)
        old_mode = self.config.stream_mode
        new_source = self.get_source_url(new_cfg)
        with self._lock:
            self.config = new_cfg
            self.db.save_config(new_cfg.model_dump())
        self.emergency_classifier.reset()
        self._snap_client = None
        if old_source != new_source or old_mode != new_cfg.stream_mode:
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

    def get_live_state(self) -> dict[str, Any]:
        stats = self.db.get_stats_state()
        system_metrics = self._get_system_metrics()
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
                "system": system_metrics,
                "speed": self.line_monitor.get_speed_summary(),
                "inference_backend": self._detector.backend if self._detector is not None else self.config.inference_backend,
                "yolo_model": self.config.yolo_model,
            }

    def _ensure_detector(self) -> ObjectDetector:
        cfg = self.config
        desired_key = (
            cfg.yolo_model,
            cfg.yolo_imgsz if cfg.inference_backend in ("auto", "openvino") else 0,
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
            tracks = detector.detect_and_track(
                frame, imgsz=cfg.yolo_imgsz, conf=cfg.confidence_threshold
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

            if step.new_violations:
                snap_base = self._get_highres_violation_frame(frame, cfg)
                sh, sw = snap_base.shape[:2]
                snap_tracks = self._scale_tracks(tracks, w, h, sw, sh)
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
                    if cfg.webhook_url.strip():
                        self._dispatch_webhook(cfg.webhook_url.strip(), rec.model_dump(), snap_path)

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

                msg = (
                    f"🚨 Red-Zone Violation: {record['vehicle_class'].upper()} #{record['track_id']} "
                    f"in {record['zone_name']} (dwell: {record['dwell_seconds']:.0f}s)"
                )
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

    def send_daily_report(
        self, webhook_url: str | None = None, date_str: str | None = None
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
                return {"ok": True, "status_code": resp.status_code, "summary": summary}
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
