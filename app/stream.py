from __future__ import annotations

import copy
import inspect
import os
import re
import shutil
import subprocess
import sys
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
from app.config import (
    RuntimeConfig,
    Settings,
    build_rtmp_url,
    build_rtsp_url,
    get_app_version,
    mask_rtsp_url,
)
from app.database import Database
from app.detector import (
    EmergencyVehicleClassifier,
    HotspotTracker,
    LineCrossingMonitor,
    MotionGate,
    ObjectDetector,
    RedZoneMonitor,
    TrackedDetection,
    _has_openvino_gpu,
    annotate_frame,
    collect_hardware_diagnostics,
    compute_optimal_input_hw,
    filter_excluded_tracks,
    format_dwell_duration,
    is_dfine_model,
    is_within_time_window,
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


class _FFmpegVAAPICapture:
    """Hardware-decoded VA-API FFmpeg pipe capture implementing OpenCV VideoCapture semantics."""

    _STREAM_DIMS_RE = re.compile(
        r"Stream #\d+:\d+.*Video:.*?\b([1-9]\d{1,3})x([1-9]\d{1,3})\b"
    )

    def __init__(
        self,
        source: str,
        render_node: str = "/dev/dri/renderD128",
        ffmpeg_cmd: list[str] | None = None,
    ) -> None:
        self.source = source
        self.render_node = render_node
        self._custom_cmd = ffmpeg_cmd
        self._release_lock = threading.Lock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_thread: threading.Thread | None = None
        self._dims_event = threading.Event()
        self._closed = False
        self._opened = False
        self._width = 0
        self._height = 0
        self._bufs: list[bytearray] = []
        self._buf_idx = 0
        self._has_prefetched = False
        self._stderr_tail: list[str] = []

    def _build_cmd(self) -> tuple[list[str], dict[str, str]]:
        env = os.environ.copy()
        env.setdefault("LIBVA_DRIVER_NAME", "iHD")
        if self._custom_cmd is not None:
            return list(self._custom_cmd), env
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "info",
            "-nostats",
            "-fflags",
            "nobuffer",
            "-flags",
            "low_delay",
        ]
        if self.source.lower().startswith("rtsp://"):
            cmd.extend(["-rtsp_transport", "tcp"])
        cmd.extend(
            [
                "-hwaccel",
                "vaapi",
                "-hwaccel_device",
                self.render_node,
                "-i",
                self.source,
                "-an",
                "-sn",
                "-dn",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "bgr24",
                "pipe:1",
            ]
        )
        return cmd, env

    def _drain_stderr(self, stderr: Any) -> None:
        try:
            for raw_line in iter(stderr.readline, b""):
                line = raw_line.decode("utf-8", errors="ignore").strip()
                if not line:
                    continue
                if len(self._stderr_tail) < 30:
                    self._stderr_tail.append(line)
                if not self._dims_event.is_set():
                    m = self._STREAM_DIMS_RE.search(line)
                    if m:
                        w, h = int(m.group(1)), int(m.group(2))
                        if 16 <= w <= 8192 and 16 <= h <= 8192:
                            self._width = w
                            self._height = h
                            self._dims_event.set()
        except Exception:
            pass
        finally:
            self._dims_event.set()

    @staticmethod
    def _read_exact_into(pipe: Any, target: bytearray) -> bool:
        view = memoryview(target)
        total = len(target)
        pos = 0
        while pos < total:
            try:
                n = pipe.readinto(view[pos:])
            except (OSError, ValueError):
                return False
            if not n:
                return False
            pos += n
        return True

    def open(self, timeout: float = 8.0) -> bool:
        cmd, env = self._build_cmd()
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,
                env=env,
            )
        except OSError:
            self.release()
            return False

        if self._proc.stderr is not None:
            self._stderr_thread = threading.Thread(
                target=self._drain_stderr,
                args=(self._proc.stderr,),
                name="ffmpeg-vaapi-stderr",
                daemon=True,
            )
            self._stderr_thread.start()
        else:
            self._dims_event.set()

        self._dims_event.wait(timeout=max(0.5, float(timeout)))
        if self._width <= 0 or self._height <= 0 or self._proc.stdout is None:
            self.release()
            return False

        nbytes = self._width * self._height * 3
        if nbytes <= 0 or nbytes > 128_000_000:
            self.release()
            return False

        self._bufs = [bytearray(nbytes), bytearray(nbytes)]
        self._buf_idx = 0
        if not self._read_exact_into(self._proc.stdout, self._bufs[0]):
            self.release()
            return False

        self._has_prefetched = True
        self._opened = True
        return True

    def isOpened(self) -> bool:
        return bool(
            self._opened
            and not self._closed
            and self._proc is not None
            and self._proc.poll() is None
        )

    def grab(self) -> bool:
        if self._closed or not self._opened or self._proc is None or self._proc.stdout is None:
            return False
        if self._has_prefetched:
            self._has_prefetched = False
            return True
        next_idx = 1 - self._buf_idx
        if not self._read_exact_into(self._proc.stdout, self._bufs[next_idx]):
            return False
        self._buf_idx = next_idx
        return True

    def retrieve(self) -> tuple[bool, np.ndarray | None]:
        if self._closed or not self._opened or self._width <= 0 or self._height <= 0:
            return (False, None)
        arr = np.frombuffer(self._bufs[self._buf_idx], dtype=np.uint8)
        expected = self._height * self._width * 3
        if arr.size != expected:
            return (False, None)
        frame = arr.reshape((self._height, self._width, 3)).copy()
        return (True, frame)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if not self.grab():
            return (False, None)
        return self.retrieve()

    def release(self) -> None:
        with self._release_lock:
            if self._closed:
                return
            self._closed = True
            self._opened = False
            proc = self._proc
            self._proc = None
        if proc is not None:
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    try:
                        proc.kill()
                        proc.wait(timeout=1.0)
                    except OSError:
                        pass
                except OSError:
                    pass
        if self._stderr_thread is not None and self._stderr_thread.is_alive():
            self._stderr_thread.join(timeout=1.0)


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
        self._config_wake_event = threading.Event()
        self._ha_wake_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._report_thread: threading.Thread | None = None
        self._ha_thread: threading.Thread | None = None

        self._detector: ObjectDetector | None = None
        self._detector_key: tuple[str, Any, str] | None = None
        self._failed_detector_key: tuple[str, Any, str] | None = None
        self._model_loading: bool = False
        self._model_loading_target: str = ""
        self._model_error: str = ""
        self.monitor = RedZoneMonitor()
        self.line_monitor = LineCrossingMonitor()
        self.hotspot = HotspotTracker(storage_path=settings.data_path / "heatmap.npy")
        self.emergency_classifier = EmergencyVehicleClassifier()
        self.motion_gate = MotionGate()
        self._motion_state: str = "off"
        self._has_inferred_frame: bool = False
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
        self._hw_diag_cache: tuple[float, dict[str, Any]] | None = None
        self._video_decode_backend: str = "software"

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

    def get_source_summary(
        self, cfg: RuntimeConfig | None = None, stream_status: str | None = None
    ) -> dict[str, str]:
        c = cfg or self.config
        status = self._stream_status if stream_status is None else stream_status
        if c.test_video_path.strip():
            fname = Path(c.test_video_path.strip()).name or "video"
            return {
                "protocol": "File",
                "stream_quality": fname,
                "source_label": f"File ({fname})",
            }
        if c.stream_mode == "http_snap" or status == "connected_http":
            return {
                "protocol": "HTTP Snap",
                "stream_quality": "",
                "source_label": "HTTP Snap",
            }
        protocol = "RTMP" if c.stream_mode == "rtmp" else "RTSP"
        quality_label = "Main Stream" if c.nvr_stream_type == "main" else "Sub Stream"
        return {
            "protocol": protocol,
            "stream_quality": quality_label,
            "source_label": f"{protocol} • {quality_label}",
        }

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
        self._config_wake_event.set()
        self._ha_wake_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        if self._report_thread:
            self._report_thread.join(timeout=2.0)
        if self._ha_thread:
            self._ha_thread.join(timeout=2.0)
        self._flush_pending_violation_webhooks(synchronous=True)
        self.hotspot.save(force=True)

    @staticmethod
    def _desired_detector_key(
        cfg: RuntimeConfig, frame_wh: tuple[int, int] | None = None
    ) -> tuple[str, tuple[int, int], str]:
        backend = (cfg.inference_backend or "auto").strip().lower()
        if is_dfine_model(cfg.yolo_model):
            hw_key = (640, 640)
        elif backend in ("onnx", "openvino", "openvino_gpu") or (
            backend == "auto" and _has_openvino_gpu()
        ):
            w, h = frame_wh if frame_wh is not None else (0, 0)
            hw_key = compute_optimal_input_hw(w, h, cfg.yolo_imgsz)
        else:
            hw_key = (0, 0)
        return (cfg.yolo_model, hw_key, backend)

    @staticmethod
    def _find_vaapi_render_node(
        dri_dir: Path = Path("/dev/dri"),
        driver_dirs: tuple[Path, ...] = (
            Path("/usr/lib/x86_64-linux-gnu/dri"),
            Path("/usr/lib/aarch64-linux-gnu/dri"),
            Path("/usr/lib/dri"),
        ),
        ffmpeg_bin: str | None = None,
        platform_name: str | None = None,
    ) -> str | None:
        if (platform_name or sys.platform) != "linux":
            return None
        has_ffmpeg = bool(shutil.which("ffmpeg") if ffmpeg_bin is None else ffmpeg_bin)
        if not has_ffmpeg:
            return None
        has_driver = any(
            (d / drv).is_file()
            for d in driver_dirs
            for drv in ("iHD_drv_video.so", "i965_drv_video.so")
        )
        if not has_driver or not dri_dir.exists():
            return None
        try:
            for entry in sorted(dri_dir.iterdir()):
                if entry.name.startswith("renderD") and os.access(entry, os.R_OK | os.W_OK):
                    return str(entry)
        except OSError:
            pass
        return None

    @staticmethod
    def _format_detector_target(
        cfg: RuntimeConfig, desired_key: tuple[str, tuple[int, int], str]
    ) -> str:
        hw = desired_key[1]
        size_str = f"{hw[1]}x{hw[0]}" if hw != (0, 0) else f"{cfg.yolo_imgsz}px"
        return f"{cfg.yolo_model} ({desired_key[2].upper()} @ {size_str})"

    def update_config(self, new_cfg: RuntimeConfig) -> None:
        old_source = self.get_source_url(self.config)
        old_mode = self.config.stream_mode
        worker_alive = self._thread is not None and self._thread.is_alive()
        with self._lock:
            eff_wh = self._resolution
            desired_key = self._desired_detector_key(new_cfg, eff_wh)
            self.config = new_cfg
            self.db.save_config(new_cfg.model_dump())
            self._failed_detector_key = None
            self._model_error = ""
            if worker_alive and (self._detector is None or desired_key != self._detector_key):
                self._model_loading = True
                self._model_loading_target = self._format_detector_target(new_cfg, desired_key)
        self.emergency_classifier.reset()
        self.motion_gate.reset()
        self._snap_client = None
        self._config_wake_event.set()
        self._ha_wake_event.set()
        new_source = self.get_source_url(new_cfg)
        if old_source != new_source or old_mode != new_cfg.stream_mode:
            if old_source != new_source:
                if self._detector is not None:
                    self._detector.reset()
                self._flush_pending_violation_webhooks(synchronous=False)
                self.hotspot.clear_tracks()
                self._has_inferred_frame = False
                with self._lock:
                    self._resolution = (0, 0)
            self._reconnect_event.set()

    def _rerender_latest_preview(self) -> None:
        with self._lock:
            raw = None if self._latest_raw_frame is None else self._latest_raw_frame.copy()
            tracks, statuses, zones, lines = self._latest_overlay
            cfg = self.config
        if raw is None:
            return
        jpg = self._render_preview_jpeg(raw, tracks, statuses, zones, lines, cfg)
        if jpg is not None:
            with self._lock:
                self._latest_jpeg = jpg

    def toggle_heatmap(self, enabled: bool | None = None, mode: str | None = None) -> bool:
        with self._lock:
            cur_on = bool(self.config.heatmap_enabled)
            cur_mode = (self.config.heatmap_mode or "all").strip().lower()
            req_mode = mode.strip().lower() if isinstance(mode, str) else None
            if req_mode == "off":
                new_val = False
                new_mode = "all"
            elif req_mode in ("all", "vehicles", "people"):
                new_val = True if enabled is None else bool(enabled)
                new_mode = req_mode
            elif enabled is not None:
                new_val = bool(enabled)
                new_mode = cur_mode if cur_mode in ("all", "vehicles", "people") else "all"
            else:
                if not cur_on:
                    new_val = True
                    new_mode = "all"
                elif cur_mode == "all":
                    new_val = True
                    new_mode = "vehicles"
                elif cur_mode == "vehicles":
                    new_val = True
                    new_mode = "people"
                else:
                    new_val = False
                    new_mode = "all"
            self.config.heatmap_enabled = new_val
            self.config.heatmap_mode = new_mode  # type: ignore[assignment]
            self.db.save_config(self.config.model_dump())
        self._rerender_latest_preview()
        return new_val

    def reset_heatmap(self) -> None:
        self.hotspot.reset(preserve_stationary=True)
        self._rerender_latest_preview()

    def get_latest_jpeg(self) -> bytes | None:
        self._last_viewer_ts = time.monotonic()
        with self._lock:
            return self._latest_jpeg

    def get_latest_raw_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._latest_raw_frame is None else self._latest_raw_frame.copy()

    @staticmethod
    def _read_cpu_temp_c(
        thermal_root: Path = Path("/sys/class/thermal"),
        hwmon_root: Path = Path("/sys/class/hwmon"),
    ) -> float | None:
        """Read host CPU temperature, preferring x86_pkg_temp / coretemp over ACPI stubs."""
        preferred_types = (
            "x86_pkg_temp",
            "cpu-thermal",
            "cpu_thermal",
            "soc_thermal",
            "k10temp",
            "coretemp",
            "tcpu",
        )
        first_valid: float | None = None
        if thermal_root.exists():
            try:
                for zone_dir in sorted(thermal_root.glob("thermal_zone*")):
                    temp_file = zone_dir / "temp"
                    if not temp_file.exists():
                        continue
                    try:
                        val = int(temp_file.read_text(encoding="utf-8").strip()) / 1000.0
                    except Exception:
                        continue
                    if not (0.0 < val < 130.0):
                        continue
                    if first_valid is None:
                        first_valid = val
                    type_file = zone_dir / "type"
                    ztype = ""
                    if type_file.exists():
                        try:
                            ztype = (
                                type_file.read_text(encoding="utf-8", errors="ignore").strip().lower()
                            )
                        except Exception:
                            pass
                    if ztype in preferred_types:
                        return round(val, 1)
            except Exception:
                pass

        if hwmon_root.exists():
            try:
                for hw_dir in sorted(hwmon_root.glob("hwmon*")):
                    name_file = hw_dir / "name"
                    temp_file = hw_dir / "temp1_input"
                    if not (name_file.exists() and temp_file.exists()):
                        continue
                    try:
                        hname = (
                            name_file.read_text(encoding="utf-8", errors="ignore").strip().lower()
                        )
                        if hname not in ("coretemp", "k10temp", "zenpower", "cpu_thermal"):
                            continue
                        val = int(temp_file.read_text(encoding="utf-8").strip()) / 1000.0
                    except Exception:
                        continue
                    if 0.0 < val < 130.0:
                        return round(val, 1)
            except Exception:
                pass

        return round(first_valid, 1) if first_valid is not None else None

    @staticmethod
    def _get_system_metrics() -> dict[str, Any]:
        """Collect host CPU, memory, process RAM, and CPU temperature."""
        cpu_percent: float | None = None
        memory_percent: float | None = None
        app_ram_mb: float | None = None

        if psutil is not None:
            try:
                cpu_percent = round(float(psutil.cpu_percent(interval=None)), 1)
                memory_percent = round(float(psutil.virtual_memory().percent), 1)
                app_ram_mb = round(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024), 1)
            except Exception:
                pass

        cpu_temp_c = StreamWorker._read_cpu_temp_c()

        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory_percent,
            "app_ram_mb": app_ram_mb,
            "cpu_temp_c": cpu_temp_c,
        }

    def build_ha_payload(self) -> dict[str, Any]:
        cfg = self.config
        stats = self.db.get_stats_state()
        speed = self.line_monitor.get_speed_summary()
        violations_total = self.db.count_violations()
        person_window_active = bool(
            cfg.person_zone_enabled
            and is_within_time_window(
                time.time(), cfg.person_zone_start_time, cfg.person_zone_end_time
            )
        )
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
            "person_alerts": int(stats.get("person_alerts", 0)),
            "person_zone_enabled": bool(cfg.person_zone_enabled),
            "person_zone_active": person_window_active,
            "person_zone_schedule": f"{cfg.person_zone_start_time} - {cfg.person_zone_end_time}",
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
        cfg = self.config
        stats = self.db.get_stats_state()
        system_metrics = self._get_system_metrics()
        ha_status = self.ha_publisher.get_status()
        person_window_active = bool(
            cfg.person_zone_enabled
            and is_within_time_window(
                time.time(), cfg.person_zone_start_time, cfg.person_zone_end_time
            )
        )
        with self._lock:
            source_summary = self.get_source_summary(self.config, self._stream_status)
            return {
                "version": get_app_version(),
                "stream_status": self._stream_status,
                "last_error": self._last_error,
                "source": mask_rtsp_url(self.get_source_url()),
                "stream_protocol": source_summary["protocol"],
                "stream_quality": source_summary["stream_quality"],
                "source_label": source_summary["source_label"],
                "resolution": {"width": self._resolution[0], "height": self._resolution[1]},
                "fps": round(self._measured_fps, 2),
                "active_cars": self._active_cars,
                "active_people": self._active_people,
                "unique_cars": stats["unique_cars"],
                "total_cars": stats["total_cars"],
                "unique_people": stats["unique_people"],
                "total_people": stats["total_people"],
                "person_alerts": int(stats.get("person_alerts", 0)),
                "person_zone_enabled": bool(cfg.person_zone_enabled),
                "person_zone_active": person_window_active,
                "person_zone_window": f"{cfg.person_zone_start_time}–{cfg.person_zone_end_time}",
                "violations_total": self.db.count_violations(),
                "counting_lines": [ln.model_dump() for ln in self.db.get_counting_lines()],
                "active_dwells": list(self._active_dwells),
                "heatmap_enabled": self.config.heatmap_enabled,
                "heatmap_mode": self.config.heatmap_mode,
                "heatmap_started_at": self.hotspot.started_at,
                "last_violation": self._last_violation,
                "last_plate": dict(self._last_plate),
                "ha": ha_status,
                "system": system_metrics,
                "speed": self.line_monitor.get_speed_summary(),
                "inference_backend": (
                    getattr(self._detector, "backend", self.config.inference_backend)
                    if self._detector is not None
                    else self.config.inference_backend
                ),
                "requested_backend": (
                    getattr(self._detector, "requested_backend", self.config.inference_backend)
                    if self._detector is not None
                    else self.config.inference_backend
                ),
                "backend_fallback": (
                    getattr(self._detector, "fallback_note", "") if self._detector is not None else ""
                ),
                "yolo_model": (
                    getattr(self._detector, "model_name", self.config.yolo_model)
                    if self._detector is not None
                    else self.config.yolo_model
                ),
                "yolo_imgsz": (
                    getattr(self._detector, "imgsz", self.config.yolo_imgsz)
                    if self._detector is not None
                    else self.config.yolo_imgsz
                ),
                "input_resolution": (
                    getattr(self._detector, "input_resolution_label", "")
                    if self._detector is not None
                    else ""
                ),
                "motion_gate_enabled": self.config.motion_gate_enabled,
                "motion_state": self._motion_state if self.config.motion_gate_enabled else "off",
                "video_decode": self._video_decode_backend,
                "model_loading": self._model_loading,
                "model_loading_target": self._model_loading_target,
                "model_error": self._model_error,
            }

    def get_hardware_diagnostics(self, refresh: bool = False) -> dict[str, Any]:
        """Collect host hardware/GPU diagnostics with a 15-second cache."""
        now_mono = time.monotonic()
        cached = self._hw_diag_cache
        if not refresh and cached is not None and (now_mono - cached[0]) < 15.0:
            base = dict(cached[1])
        else:
            base = collect_hardware_diagnostics(run_clinfo=True)
            self._hw_diag_cache = (now_mono, dict(base))

        with self._lock:
            det = self._detector
            detector_runtime = {
                "requested_backend": (
                    getattr(det, "requested_backend", self.config.inference_backend)
                    if det is not None
                    else self.config.inference_backend
                ),
                "active_backend": (
                    getattr(det, "backend", self.config.inference_backend)
                    if det is not None
                    else self.config.inference_backend
                ),
                "active_device": getattr(det, "device", "") if det is not None else "",
                "backend_fallback": getattr(det, "fallback_note", "") if det is not None else "",
                "yolo_model": (
                    getattr(det, "model_name", self.config.yolo_model)
                    if det is not None
                    else self.config.yolo_model
                ),
                "input_resolution": (
                    getattr(det, "input_resolution_label", "") if det is not None else ""
                ),
                "video_decode": self._video_decode_backend,
            }
        base["detector_runtime"] = detector_runtime
        base["cpu_temp_c"] = self._read_cpu_temp_c()
        return base

    def _ensure_detector(self, frame_wh: tuple[int, int] | None = None) -> ObjectDetector | None:
        cfg = self.config
        with self._lock:
            eff_wh = (
                (int(frame_wh[0]), int(frame_wh[1]))
                if (frame_wh is not None and frame_wh[0] > 0 and frame_wh[1] > 0)
                else self._resolution
            )
        desired_key = self._desired_detector_key(cfg, eff_wh)
        if self._detector is not None and self._detector_key == desired_key:
            if not is_dfine_model(cfg.yolo_model) and getattr(self._detector, "backend", "") not in (
                "onnx",
                "openvino",
                "openvino_gpu",
            ):
                self._detector.imgsz = cfg.yolo_imgsz
                if eff_wh[0] > 0 and eff_wh[1] > 0:
                    self._detector.input_hw = compute_optimal_input_hw(
                        eff_wh[0], eff_wh[1], cfg.yolo_imgsz
                    )
                else:
                    self._detector.input_hw = (cfg.yolo_imgsz, cfg.yolo_imgsz)
            with self._lock:
                self._model_loading = False
                self._model_loading_target = ""
            return self._detector

        if self._failed_detector_key == desired_key:
            with self._lock:
                self._model_loading = False
                self._model_loading_target = ""
            return self._detector

        prev_detector = self._detector
        with self._lock:
            prev_status = self._stream_status
            prev_error = self._last_error
            self._model_loading = True
            self._model_loading_target = self._format_detector_target(cfg, desired_key)
            self._model_error = ""
            if prev_status != "loading_model":
                self._stream_status = "loading_model"
                self._last_error = ""
            if prev_detector is None:
                self._latest_jpeg = self._render_placeholder(f"Loading {cfg.yolo_model}...")

        det_kwargs: dict[str, Any] = {
            "model_name": cfg.yolo_model,
            "imgsz": cfg.yolo_imgsz,
            "inference_backend": cfg.inference_backend,
            "models_dir": self.settings.data_path / "models",
        }
        try:
            if desired_key[1] != (0, 0) and "input_hw" in inspect.signature(ObjectDetector).parameters:
                det_kwargs["input_hw"] = desired_key[1]
        except (TypeError, ValueError):
            pass

        try:
            new_detector = ObjectDetector(**det_kwargs)
        except Exception as exc:
            with self._lock:
                self._failed_detector_key = desired_key
                self._model_error = f"Failed to load {cfg.yolo_model}: {exc}"
                if self._desired_detector_key(self.config, eff_wh) == desired_key:
                    self._model_loading = False
                    self._model_loading_target = ""
                if prev_detector is not None:
                    self._stream_status = prev_status
                    self._last_error = prev_error
                else:
                    self._stream_status = "error"
                    self._last_error = f"Failed to load YOLO model: {exc}"
                    self._latest_jpeg = self._render_placeholder(f"Model error: {exc}")
            return prev_detector

        if prev_detector is not None:
            prev_detector.stabilizer.clear_raw_bindings()
            new_detector.stabilizer = prev_detector.stabilizer
        if (
            not is_dfine_model(cfg.yolo_model)
            and eff_wh[0] > 0
            and eff_wh[1] > 0
            and desired_key[1] == (0, 0)
        ):
            new_detector.input_hw = compute_optimal_input_hw(eff_wh[0], eff_wh[1], cfg.yolo_imgsz)

        self.motion_gate.reset()
        self._has_inferred_frame = False
        with self._lock:
            self._detector = new_detector
            self._detector_key = desired_key
            self._failed_detector_key = None
            self._model_error = ""
            if self._desired_detector_key(self.config, eff_wh) == desired_key:
                self._model_loading = False
                self._model_loading_target = ""
            if prev_status in ("connected", "connected_http"):
                self._stream_status = prev_status
                self._last_error = prev_error
            elif prev_detector is None or self._stream_status == "loading_model":
                self._stream_status = "initializing" if prev_detector is None else prev_status
                self._last_error = "" if prev_detector is None else prev_error

        self._ha_wake_event.set()
        return new_detector

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

    @staticmethod
    def _render_subject_snapshot(
        base_frame: np.ndarray,
        tracks: list[TrackedDetection],
        track_statuses: dict[int, dict[str, Any]],
        subject_tid: int,
        *,
        dwell_threshold: float,
        annotation_scale: float,
        person_dwell_threshold: float = 2.0,
    ) -> np.ndarray:
        """Render a clean snapshot with only the offending subject's bounding box and label."""
        tid = int(subject_tid)
        subject_tracks = [t for t in tracks if int(t.track_id) == tid]
        subject_statuses = {tid: dict(track_statuses[tid])} if tid in track_statuses else {}
        return annotate_frame(
            frame=base_frame,
            tracks=subject_tracks,
            track_statuses=subject_statuses,
            zones=[],
            dwell_threshold=dwell_threshold,
            hotspot=None,
            show_heatmap=False,
            counting_lines=[],
            annotation_scale=annotation_scale,
            person_dwell_threshold=person_dwell_threshold,
            draw_anchor_dots=False,
        )

    def _refresh_violation_snapshot(self, entry: dict[str, Any], dwell_seconds: float) -> bool:
        """Re-render the violation snapshot using the latest in-zone frame and parked dwell duration."""
        latest_clean = entry.get("latest_clean_frame")
        if isinstance(latest_clean, np.ndarray) and latest_clean.size > 0:
            base_img = latest_clean.copy()
        else:
            clean_bytes: bytes = entry.get("clean_jpeg", b"")
            if not clean_bytes:
                return False
            base_img = cv2.imdecode(np.frombuffer(clean_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if base_img is None:
                return False

        orig_tid = int(entry["orig_track_id"])
        statuses = {k: dict(val) for k, val in entry["track_statuses"].items()}
        is_person = bool(entry.get("is_person")) or str(entry.get("zone_id", "")).startswith("person_zone")
        default_st = (
            {
                "in_red_zone": False,
                "dwell_seconds": 0.0,
                "violation": False,
                "zone_name": None,
                "emergency": False,
                "in_person_zone": True,
                "person_dwell_seconds": round(float(dwell_seconds), 1),
                "person_violation": True,
            }
            if is_person
            else {
                "in_red_zone": True,
                "dwell_seconds": round(float(dwell_seconds), 1),
                "violation": True,
                "zone_name": None,
                "emergency": False,
            }
        )
        target_status = statuses.setdefault(orig_tid, default_st)
        if is_person:
            target_status["in_person_zone"] = True
            target_status["person_violation"] = True
            target_status["person_dwell_seconds"] = round(float(dwell_seconds), 1)
        else:
            target_status["in_red_zone"] = True
            target_status["violation"] = True
            target_status["dwell_seconds"] = round(float(dwell_seconds), 1)
        plate_txt = str((entry.get("record") or {}).get("license_plate") or "").strip()
        if plate_txt and not target_status.get("license_plate"):
            target_status["license_plate"] = plate_txt

        updated_img = self._render_subject_snapshot(
            base_frame=base_img,
            tracks=entry["tracks"],
            track_statuses={orig_tid: target_status},
            subject_tid=orig_tid,
            dwell_threshold=float(entry["dwell_threshold"]),
            annotation_scale=float(entry["annotation_scale"]),
            person_dwell_threshold=float(entry.get("person_dwell_threshold", 2.0)),
        )
        return self._write_jpeg_atomic(entry["snap_path"], updated_img, quality=92)

    def _finalize_violation_snap_entry(
        self,
        snap_entry: dict[str, Any],
        dwell_seconds: float | None = None,
        *,
        synchronous_webhook: bool = False,
    ) -> None:
        final_dwell = round(
            float(
                dwell_seconds
                if dwell_seconds is not None
                else snap_entry.get("last_rendered_dwell", 0.0)
            ),
            1,
        )
        last_seen_iso = str(snap_entry.get("last_seen_iso") or "")
        plate_txt = str((snap_entry.get("record") or {}).get("license_plate") or "")
        self.db.update_violation_dwell(
            snap_entry["violation_id"],
            final_dwell,
            flagged_at=last_seen_iso or None,
            license_plate=plate_txt or None,
        )
        self._refresh_violation_snapshot(snap_entry, final_dwell)
        snap_entry["last_rendered_dwell"] = final_dwell
        with self._lock:
            if (
                self._last_violation is not None
                and self._last_violation.get("id") == snap_entry["violation_id"]
            ):
                self._last_violation["dwell_seconds"] = final_dwell
                if last_seen_iso:
                    self._last_violation["flagged_at"] = last_seen_iso
                if plate_txt:
                    self._last_violation["license_plate"] = plate_txt
        if snap_entry.get("webhook_pending"):
            snap_entry["webhook_pending"] = False
            wh_url = (
                self.config.webhook_url.strip()
                or str(snap_entry.get("webhook_url") or "").strip()
            )
            if wh_url:
                rec_payload = dict(snap_entry.get("record") or {})
                rec_payload["dwell_seconds"] = final_dwell
                if last_seen_iso:
                    rec_payload["flagged_at"] = last_seen_iso
                if plate_txt:
                    rec_payload["license_plate"] = plate_txt
                self._dispatch_webhook(
                    wh_url,
                    rec_payload,
                    snap_entry["snap_path"],
                    synchronous=synchronous_webhook,
                )

    def _flush_pending_violation_webhooks(self, *, synchronous: bool = False) -> None:
        with self._lock:
            pending_entries = list(self._active_violation_snaps.values())
            self._active_violation_snaps.clear()
        for entry in pending_entries:
            self._finalize_violation_snap_entry(entry, synchronous_webhook=synchronous)

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
        target_w = min(max(cfg.preview_max_width, 960), 1920) if cfg.preview_max_width > 0 else w
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
            person_dwell_threshold=cfg.person_zone_dwell_seconds,
            heatmap_mode=cfg.heatmap_mode,
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

    def _handle_step_events(
        self,
        frame: np.ndarray,
        w: int,
        h: int,
        tracks: list[TrackedDetection],
        step: Any,
        zones: list[Any],
        counting_lines: list[Any],
        cfg: RuntimeConfig,
        speeding_events: list[dict[str, Any]],
        *,
        now_ts: float | None = None,
        inferred_this_step: bool = True,
    ) -> None:
        if step.new_violations or speeding_events:
            snap_base = self._get_highres_violation_frame(frame, cfg)
            sh, sw = snap_base.shape[:2]
            snap_tracks = self._scale_tracks(tracks, w, h, sw, sh)
            track_bbox_map = {t.track_id: t.bbox_xyxy for t in snap_tracks}
            plates_by_tid: dict[int, tuple[str, float]] = {}

            if cfg.alpr_enabled:
                event_tids = {
                    v.track_id for v in step.new_violations if v.vehicle_class != "person"
                } | {int(sp["track_id"]) for sp in speeding_events}
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
            clean_source = snap_base
            if step.new_violations:
                ok_clean, clean_buf = cv2.imencode(
                    ".jpg", clean_source, [cv2.IMWRITE_JPEG_QUALITY, 94]
                )
                if ok_clean:
                    clean_jpeg = clean_buf.tobytes()

            evicted_entries: list[dict[str, Any]] = []
            for v in step.new_violations:
                is_person_alert = v.vehicle_class == "person" or str(v.zone_id).startswith(
                    "person_zone"
                )
                v_tid = int(v.track_id)
                ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                fname_prefix = "person" if is_person_alert else "violation"
                fname = f"{fname_prefix}_{ts_slug}_t{v.track_id}.jpg"
                snap_path = self.snapshots_dir / fname
                v_tracks = [copy.deepcopy(t) for t in snap_tracks if int(t.track_id) == v_tid]
                v_statuses = {v_tid: copy.deepcopy(step.track_statuses.get(v_tid, {}))}
                v_img = self._render_subject_snapshot(
                    base_frame=snap_base,
                    tracks=v_tracks,
                    track_statuses=v_statuses,
                    subject_tid=v_tid,
                    dwell_threshold=cfg.dwell_seconds,
                    annotation_scale=cfg.annotation_scale,
                    person_dwell_threshold=cfg.person_zone_dwell_seconds,
                )
                self._write_jpeg_atomic(snap_path, v_img, quality=92)
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
                rec_dict = rec.model_dump()
                if clean_jpeg:
                    now_mono = time.monotonic()
                    snap_entry: dict[str, Any] = {
                        "violation_id": rec.id,
                        "orig_track_id": v_tid,
                        "zone_id": v.zone_id,
                        "is_person": is_person_alert,
                        "snap_path": snap_path,
                        "clean_jpeg": clean_jpeg,
                        "latest_clean_frame": clean_source.copy(),
                        "clean_w": int(sw),
                        "tracks": v_tracks,
                        "track_statuses": v_statuses,
                        "zones": [],
                        "counting_lines": [],
                        "dwell_threshold": float(cfg.dwell_seconds),
                        "person_dwell_threshold": float(cfg.person_zone_dwell_seconds),
                        "annotation_scale": float(cfg.annotation_scale),
                        "created_ts": now_mono,
                        "last_update_ts": now_mono,
                        "last_highres_ts": now_mono,
                        "last_rendered_dwell": round(float(v.dwell_seconds), 1),
                        "last_seen_iso": v.flagged_at,
                        "record": dict(rec_dict),
                        "webhook_url": cfg.webhook_url.strip(),
                        "webhook_pending": not is_person_alert,
                    }
                    with self._lock:
                        expired_keys = [
                            k
                            for k, val in self._active_violation_snaps.items()
                            if (now_mono - float(val["created_ts"])) > 14400.0
                        ]
                        for ek in expired_keys:
                            popped = self._active_violation_snaps.pop(ek, None)
                            if popped is not None:
                                evicted_entries.append(popped)
                        while len(self._active_violation_snaps) >= 16:
                            oldest_key = min(
                                self._active_violation_snaps,
                                key=lambda k: float(self._active_violation_snaps[k]["created_ts"]),
                            )
                            popped = self._active_violation_snaps.pop(oldest_key, None)
                            if popped is not None:
                                evicted_entries.append(popped)
                        self._active_violation_snaps[(v_tid, v.zone_id)] = snap_entry
                with self._lock:
                    self._last_violation = rec_dict
                if cfg.webhook_url.strip() and (is_person_alert or not clean_jpeg):
                    self._dispatch_webhook(cfg.webhook_url.strip(), rec_dict, snap_path)
                if cfg.ha_enabled:
                    ha_event_type = (
                        "car_tracker_person_alert" if is_person_alert else "car_tracker_violation"
                    )
                    threading.Thread(
                        target=self.ha_publisher.fire_event,
                        args=(ha_event_type, rec_dict, cfg.ha_url, cfg.ha_token),
                        daemon=True,
                    ).start()

            for ev in evicted_entries:
                self._finalize_violation_snap_entry(ev)

            for sp in speeding_events:
                sp_tid = int(sp["track_id"])
                now_iso = datetime.now(timezone.utc).isoformat()
                ts_slug = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                fname = f"speeding_{ts_slug}_t{sp['track_id']}.jpg"
                snap_path = self.snapshots_dir / fname
                plate_text = plates_by_tid.get(sp_tid, ("", 0.0))[0]
                sp_status = dict(step.track_statuses.get(sp_tid) or {})
                sp_status["speed_mph"] = float(sp["speed_mph"])
                if plate_text:
                    sp_status["license_plate"] = plate_text
                sp_img = self._render_subject_snapshot(
                    base_frame=snap_base,
                    tracks=snap_tracks,
                    track_statuses={sp_tid: sp_status},
                    subject_tid=sp_tid,
                    dwell_threshold=cfg.dwell_seconds,
                    annotation_scale=cfg.annotation_scale,
                    person_dwell_threshold=cfg.person_zone_dwell_seconds,
                )
                self._write_jpeg_atomic(snap_path, sp_img, quality=92)
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
            eff_now_ts = float(now_ts) if now_ts is not None else time.time()
            any_completed = False
            track_ids_present = {t.track_id for t in tracks}
            for upd in step.dwell_updates:
                with self._lock:
                    snap_entry = self._active_violation_snaps.get(upd.violation_key)
                if snap_entry is None:
                    continue
                rounded_dwell = round(float(upd.dwell_seconds), 1)
                if upd.completed:
                    with self._lock:
                        popped_entry = self._active_violation_snaps.pop(upd.violation_key, None)
                    if popped_entry is not None:
                        self._finalize_violation_snap_entry(popped_entry, rounded_dwell)
                    any_completed = True
                    continue

                should_refresh = (now_mono - float(snap_entry["last_update_ts"])) >= 5.0 and int(
                    round(rounded_dwell)
                ) != int(round(float(snap_entry["last_rendered_dwell"])))

                orig_tid = int(snap_entry["orig_track_id"])
                zone_flag_key = "in_person_zone" if snap_entry.get("is_person") else "in_red_zone"
                in_zone_active = (
                    inferred_this_step
                    and orig_tid in track_ids_present
                    and bool((step.track_statuses.get(orig_tid) or {}).get(zone_flag_key))
                )
                if in_zone_active:
                    snap_entry["last_seen_iso"] = (
                        datetime.fromtimestamp(eff_now_ts, tz=timezone.utc).astimezone().isoformat()
                    )
                    fresh_base: np.ndarray | None = None
                    if w >= int(snap_entry.get("clean_w", 0)):
                        fresh_base = frame
                    elif (
                        should_refresh
                        and (now_mono - float(snap_entry.get("last_highres_ts", 0.0))) >= 15.0
                    ):
                        cand = self._get_highres_violation_frame(frame, cfg)
                        snap_entry["last_highres_ts"] = now_mono
                        if cand.shape[1] >= int(snap_entry.get("clean_w", 0)):
                            fresh_base = cand

                    if fresh_base is not None:
                        fh, fw = fresh_base.shape[:2]
                        snap_entry["latest_clean_frame"] = fresh_base.copy()
                        snap_entry["clean_w"] = int(fw)
                        scaled_t = (
                            tracks
                            if (fw == w and fh == h)
                            else self._scale_tracks(tracks, w, h, fw, fh)
                        )
                        subject_tracks = [
                            copy.deepcopy(t) for t in scaled_t if int(t.track_id) == orig_tid
                        ]
                        snap_entry["tracks"] = subject_tracks
                        existing_plate = str(
                            (snap_entry.get("record") or {}).get("license_plate")
                            or (snap_entry.get("track_statuses", {}).get(orig_tid) or {}).get(
                                "license_plate"
                            )
                            or ""
                        )
                        if (
                            not existing_plate
                            and should_refresh
                            and cfg.alpr_enabled
                            and not snap_entry.get("is_person")
                        ):
                            t_bbox = next(
                                (t.bbox_xyxy for t in subject_tracks), None
                            )
                            plate, plate_conf = self.alpr.read_plate(fresh_base, t_bbox)
                            if plate:
                                existing_plate = plate
                                snap_entry.setdefault("record", {})["license_plate"] = plate
                                with self._lock:
                                    self._last_plate = {
                                        "plate": plate,
                                        "confidence": plate_conf,
                                        "updated_at": datetime.now(timezone.utc).isoformat(),
                                    }
                        orig_st = copy.deepcopy(step.track_statuses.get(orig_tid) or {})
                        if existing_plate:
                            orig_st["license_plate"] = existing_plate
                        snap_entry["track_statuses"] = {orig_tid: orig_st}
                        snap_entry["zones"] = []
                        snap_entry["counting_lines"] = []
                        snap_entry["annotation_scale"] = float(cfg.annotation_scale)

                if should_refresh:
                    last_seen_iso = str(snap_entry.get("last_seen_iso") or "")
                    plate_txt = str((snap_entry.get("record") or {}).get("license_plate") or "")
                    self.db.update_violation_dwell(
                        snap_entry["violation_id"],
                        rounded_dwell,
                        flagged_at=last_seen_iso or None,
                        license_plate=plate_txt or None,
                    )
                    self._refresh_violation_snapshot(snap_entry, rounded_dwell)
                    snap_entry["last_update_ts"] = now_mono
                    snap_entry["last_rendered_dwell"] = rounded_dwell
                    with self._lock:
                        if (
                            self._last_violation is not None
                            and self._last_violation.get("id") == snap_entry["violation_id"]
                        ):
                            self._last_violation["dwell_seconds"] = rounded_dwell
                            if last_seen_iso:
                                self._last_violation["flagged_at"] = last_seen_iso
                            if plate_txt:
                                self._last_violation["license_plate"] = plate_txt
            if any_completed:
                self._ha_wake_event.set()

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

            with self._lock:
                prev_tracks, prev_statuses, _, _ = self._latest_overlay
                has_active_dwells = bool(self._active_dwells)
                prev_active_cars = self._active_cars
                prev_active_people = self._active_people
                prev_res = self._resolution
                has_inferred = self._has_inferred_frame

            has_unconfirmed = any(t.track_id not in self.monitor._counted_tracks for t in prev_tracks)
            res_changed = prev_res != (0, 0) and prev_res != (int(w), int(h))

            if cfg.motion_gate_enabled:
                gate_infer, motion_state = self.motion_gate.should_infer(
                    frame, zones=zones, idle_interval=cfg.motion_idle_interval
                )
            else:
                gate_infer, motion_state = True, "off"

            run_infer = gate_infer or (not has_inferred) or has_unconfirmed or res_changed

            if run_infer:
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
                self.hotspot.update(
                    tracks,
                    w,
                    h,
                    movement_threshold=cfg.movement_threshold,
                    now_ts=now_ts,
                )

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
                    person_zone_enabled=cfg.person_zone_enabled,
                    person_zone_start_time=cfg.person_zone_start_time,
                    person_zone_end_time=cfg.person_zone_end_time,
                    person_zone_dwell_seconds=cfg.person_zone_dwell_seconds,
                    person_zone_scope=cfg.person_zone_scope,
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
                    or step.new_person_alerts
                ):
                    self.db.increment_stats(
                        unique_cars=step.new_unique_cars,
                        total_cars=step.new_total_cars,
                        unique_people=step.new_unique_people,
                        total_people=step.new_total_people,
                        person_alerts=step.new_person_alerts,
                    )

                new_speeds = self.line_monitor.consume_new_speeds()
                speeding_events = (
                    [sp for sp in new_speeds if sp["speed_mph"] > cfg.speed_limit_mph]
                    if cfg.speed_capture_enabled
                    else []
                )
                self._handle_step_events(
                    frame=frame,
                    w=w,
                    h=h,
                    tracks=tracks,
                    step=step,
                    zones=zones,
                    counting_lines=counting_lines,
                    cfg=cfg,
                    speeding_events=speeding_events,
                    now_ts=now_ts,
                    inferred_this_step=True,
                )
                track_statuses = step.track_statuses
                active_cars = step.active_cars
                active_people = step.active_people
                self._has_inferred_frame = True
            else:
                tracks = filter_excluded_tracks(list(prev_tracks), w, h, zones)
                track_statuses = dict(prev_statuses)
                active_cars = prev_active_cars
                active_people = prev_active_people
                has_red_zones = any(
                    z.enabled and getattr(z, "zone_type", "red_zone") == "red_zone" for z in zones
                )
                if tracks and (has_active_dwells or has_red_zones or cfg.person_zone_enabled):
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
                        person_zone_enabled=cfg.person_zone_enabled,
                        person_zone_start_time=cfg.person_zone_start_time,
                        person_zone_end_time=cfg.person_zone_end_time,
                        person_zone_dwell_seconds=cfg.person_zone_dwell_seconds,
                        person_zone_scope=cfg.person_zone_scope,
                    )
                    if step.new_person_alerts:
                        self.db.increment_stats(person_alerts=step.new_person_alerts)
                    track_speeds = self.line_monitor.get_track_speeds()
                    for tid, mph in track_speeds.items():
                        if tid in step.track_statuses:
                            step.track_statuses[tid]["speed_mph"] = mph
                    self._handle_step_events(
                        frame=frame,
                        w=w,
                        h=h,
                        tracks=tracks,
                        step=step,
                        zones=zones,
                        counting_lines=counting_lines,
                        cfg=cfg,
                        speeding_events=[],
                        now_ts=now_ts,
                        inferred_this_step=False,
                    )
                    track_statuses = step.track_statuses
                    active_cars = step.active_cars
                    active_people = step.active_people

            jpg_bytes = (
                self._render_preview_jpeg(
                    frame=frame,
                    tracks=tracks,
                    track_statuses=track_statuses,
                    zones=zones,
                    counting_lines=counting_lines,
                    cfg=cfg,
                )
                if publish_preview
                else None
            )

            dwells = [
                {"track_id": tid, **st}
                for tid, st in track_statuses.items()
                if st.get("in_red_zone")
            ]

            with self._lock:
                self._latest_raw_frame = frame
                self._latest_overlay = (list(tracks), dict(track_statuses), list(zones), list(counting_lines))
                if jpg_bytes is not None:
                    self._latest_jpeg = jpg_bytes
                    self._last_preview_ts = time.monotonic()
                self._resolution = (int(w), int(h))
                self._measured_fps = 1.0 / max(dt, 1e-3)
                self._motion_state = motion_state
                self._active_cars = active_cars
                self._active_people = active_people
                self._active_dwells = dwells

            red_zone_occupied = len(dwells) > 0
            if red_zone_occupied != self._prev_red_zone_occupied:
                self._prev_red_zone_occupied = red_zone_occupied
                self._ha_wake_event.set()
        except Exception as exc:
            self._set_status("error", f"Inference error: {exc}")
            time.sleep(1.0)

    def _run_http_snap_fallback(self) -> bool:
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
                detector = self._ensure_detector(frame_wh=(frame.shape[1], frame.shape[0]))
                if detector is None:
                    self._config_wake_event.wait(timeout=1.0)
                    self._config_wake_event.clear()
                    continue
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

    def _dispatch_webhook(
        self,
        url: str,
        record: dict[str, Any],
        snap_path: Path,
        *,
        synchronous: bool = False,
    ) -> None:
        jpg_bytes: bytes | None = None
        if "discord.com/api/webhooks" in url and snap_path.exists():
            try:
                jpg_bytes = snap_path.read_bytes()
            except OSError:
                jpg_bytes = None

        def _send() -> None:
            try:
                import requests

                dwell_txt = format_dwell_duration(float(record.get("dwell_seconds", 0.0)))
                if record.get("zone_id") == "speeding":
                    msg = (
                        f"🚨 Speeding Alert: {record['vehicle_class'].upper()} #{record['track_id']} "
                        f"— {record['zone_name']}"
                    )
                elif record.get("vehicle_class") == "person" or str(
                    record.get("zone_id", "")
                ).startswith("person_zone"):
                    msg = (
                        f"🚶 After-Hours Person Alert: PERSON #{record['track_id']} "
                        f"in {record['zone_name']} (dwell: {dwell_txt})"
                    )
                else:
                    msg = (
                        f"🚨 Red-Zone Violation: {record['vehicle_class'].upper()} #{record['track_id']} "
                        f"in {record['zone_name']} (dwell: {dwell_txt})"
                    )
                if record.get("license_plate"):
                    msg = f"{msg} • 🪪 Plate: **{record['license_plate']}**"
                if "discord.com/api/webhooks" in url and jpg_bytes is not None:
                    requests.post(
                        url,
                        data={"content": msg},
                        files={"file": (snap_path.name, jpg_bytes, "image/jpeg")},
                        timeout=8,
                    )
                else:
                    requests.post(url, json={"text": msg, "content": msg, "violation": record}, timeout=8)
            except Exception:
                pass

        if synchronous:
            _send()
        else:
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
        lines_out.append(f"• **Person Zone Alerts Today:** {summary.get('person_alerts_today', 0)}")
        counting_lines = [
            ln for ln in (summary.get("counting_lines") or []) if ln.get("line_role", "both") != "speed"
        ]
        if counting_lines:
            lines_out.append("• **Traffic Flow Lines (total A→B / B→A):**")
            for ln in counting_lines:
                a_b = int(ln.get("count_a_to_b", 0))
                b_a = int(ln.get("count_b_to_a", 0))
                lines_out.append(f"  - {ln['name']}: **{a_b + b_a}** ({a_b} / {b_a})")
        cum = summary.get("cumulative_stats") or {}
        lines_out.append(
            f"• **All-Time Totals:** {cum.get('unique_cars', 0)} unique vehicles, "
            f"{cum.get('unique_people', 0)} unique people, "
            f"{cum.get('person_alerts', 0)} person alerts"
        )
        return "\n".join(lines_out)

    def reset_all_counts(self) -> None:
        """Reset cumulative vehicle/people counts, hourly traffic, line crossings, speed history, and heatmap."""
        self.db.reset_stats()
        self.db.reset_line_counts()
        self.monitor.reset()
        self.line_monitor.reset()
        self.hotspot.reset(preserve_stationary=True)
        self._flush_pending_violation_webhooks(synchronous=False)
        self._rerender_latest_preview()
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

    def _consume_live_capture_loop(
        self,
        cap: Any,
        rtsp_state: dict[str, Any],
        rtsp_lock: threading.Lock,
        reader_stop: threading.Event,
    ) -> None:
        """Drain live capture via grab() and decode pixels only when inference or an active viewer needs them."""
        last_retrieve_ts = 0.0
        has_grab = hasattr(cap, "grab") and hasattr(cap, "retrieve")
        try:
            while (
                not reader_stop.is_set()
                and not self._stop_event.is_set()
                and not self._reconnect_event.is_set()
            ):
                if not has_grab:
                    ok_r, frm = cap.read()
                    if not ok_r or frm is None:
                        rtsp_state["alive"] = False
                        break
                    last_retrieve_ts = time.monotonic()
                    with rtsp_lock:
                        rtsp_state["frame"] = frm
                        rtsp_state["seq"] += 1
                    continue

                if not cap.grab():
                    rtsp_state["alive"] = False
                    break
                now_r = time.monotonic()
                viewer_active = (now_r - getattr(self, "_last_viewer_ts", 0.0)) <= 5.0
                motion_idle = bool(
                    self.config.motion_gate_enabled
                    and self._motion_state == "idle"
                    and not self._active_dwells
                )
                infer_period = 1.0 / max(self.config.target_fps, 0.5)
                min_period = (
                    min(infer_period, 0.40 if motion_idle else 0.08)
                    if viewer_active
                    else infer_period
                )
                if last_retrieve_ts > 0.0 and (now_r - last_retrieve_ts) < (min_period * 0.85):
                    continue
                ok_r, frm = cap.retrieve()
                if not ok_r or frm is None:
                    rtsp_state["alive"] = False
                    break
                last_retrieve_ts = now_r
                with rtsp_lock:
                    rtsp_state["frame"] = frm
                    rtsp_state["seq"] += 1
        finally:
            cap.release()

    def _run_loop(self) -> None:
        with self._lock:
            self._latest_jpeg = self._render_placeholder("Loading YOLO model...")

        backoff = 1.0
        vaapi_failed_source = ""
        while not self._stop_event.is_set():
            self._reconnect_event.clear()
            self._config_wake_event.clear()
            cfg = self.config

            source = self.get_source_url(cfg)
            is_file = bool(cfg.test_video_path.strip())

            if not is_file and not cfg.nvr_pass and not cfg.nvr_user:
                self._video_decode_backend = "software"
                self._set_status("waiting_for_credentials", "Enter NVR username/password in Settings.")
                with self._lock:
                    self._latest_jpeg = self._render_placeholder("Configure NVR credentials or Test Video in Settings")
                self._config_wake_event.wait(timeout=2.0)
                continue

            if not is_file and cfg.stream_mode == "http_snap":
                self._video_decode_backend = "software"
                if self._run_http_snap_fallback():
                    backoff = 1.0
                    continue
                self._set_status("reconnecting", f"HTTP Snap failed for {cfg.nvr_host}")
                with self._lock:
                    self._latest_jpeg = self._render_placeholder(f"Reconnecting HTTP Snap to {cfg.nvr_host}...")
                self._config_wake_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, 15.0)
                continue

            self._set_status("connecting", "")
            with self._lock:
                self._latest_jpeg = self._render_placeholder(f"Connecting to {mask_rtsp_url(source)}...")

            cap: Any = None
            self._video_decode_backend = "software"
            if not is_file and source != vaapi_failed_source:
                render_node = self._find_vaapi_render_node()
                if render_node is not None:
                    va_cap = _FFmpegVAAPICapture(source, render_node=render_node)
                    if va_cap.open():
                        cap = va_cap
                        self._video_decode_backend = "vaapi"
                    else:
                        va_cap.release()
                        vaapi_failed_source = source

            if cap is None:
                cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                self._video_decode_backend = "software"
                # Fall back to Reolink HTTP Snap API when RTSP cannot open
                if not is_file and cfg.stream_mode == "auto" and self._run_http_snap_fallback():
                    backoff = 1.0
                    continue
                self._set_status("reconnecting", f"Cannot open {mask_rtsp_url(source)}")
                with self._lock:
                    self._latest_jpeg = self._render_placeholder(f"Reconnecting to {mask_rtsp_url(source)}...")
                self._config_wake_event.wait(timeout=backoff)
                backoff = min(backoff * 2.0, 15.0)
                continue

            if hasattr(cap, "set"):
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            backoff = 1.0
            last_infer_ts = 0.0
            first_frame_seen = False

            # Dedicated RTSP/RTMP reader thread drains cap.grab() at stream rate and skips unneeded retrieve().
            # A separate preview thread encodes only the newest frame strictly forward in time (no rubber-banding).
            rtsp_lock = threading.Lock()
            rtsp_state: dict[str, Any] = {"frame": None, "seq": 0, "alive": True}
            reader_stop = threading.Event()
            reader_thread: threading.Thread | None = None
            preview_thread: threading.Thread | None = None
            if not is_file:
                def _rtsp_reader() -> None:
                    self._consume_live_capture_loop(cap, rtsp_state, rtsp_lock, reader_stop)

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
                            motion_idle = bool(
                                pcfg.motion_gate_enabled
                                and self._motion_state == "idle"
                                and not self._active_dwells
                            )
                        jpg = self._render_preview_jpeg(frm, tracks, statuses, zones, lines, pcfg)
                        if jpg is not None:
                            with self._lock:
                                self._latest_jpeg = jpg
                        if motion_idle:
                            for _ in range(8):
                                if (
                                    reader_stop.is_set()
                                    or self._stop_event.is_set()
                                    or self._reconnect_event.is_set()
                                    or self._motion_state != "idle"
                                    or self._active_dwells
                                ):
                                    break
                                time.sleep(0.05)
                        else:
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
                        if self._detector is not None:
                            self._detector.reset()
                        self._flush_pending_violation_webhooks(synchronous=False)
                        self.hotspot.clear_tracks()
                        self.motion_gate.reset()
                        self._has_inferred_frame = False
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

                detector = self._ensure_detector(frame_wh=(frame.shape[1], frame.shape[0]))
                if detector is None:
                    self._config_wake_event.wait(timeout=1.0)
                    self._config_wake_event.clear()
                    continue

                if not first_frame_seen:
                    first_frame_seen = True
                    self._set_status("connected", "")

                t0 = time.monotonic()
                dt = t0 - last_infer_ts if last_infer_ts > 0 else interval
                last_infer_ts = t0
                self._process_single_frame(frame, detector, cfg, dt, publish_preview=is_file)

            reader_stop.set()
            if isinstance(cap, _FFmpegVAAPICapture):
                cap.release()
            if preview_thread is not None:
                preview_thread.join(timeout=1.5)
            if reader_thread is not None:
                reader_thread.join(timeout=3.0)
            else:
                cap.release()
            if not first_frame_seen and not is_file and cfg.stream_mode == "auto" and not self._stop_event.is_set() and not self._reconnect_event.is_set():
                self._run_http_snap_fallback()
