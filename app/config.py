from __future__ import annotations

from pathlib import Path
from typing import Literal
from urllib.parse import quote

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    nvr_host: str = "192.168.68.51"
    nvr_port: int = 554
    nvr_channel: int = 7
    nvr_stream_type: Literal["main", "sub"] = "main"
    stream_mode: Literal["auto", "rtmp", "http_snap", "rtsp"] = "auto"
    nvr_user: str = "admin"
    nvr_pass: str = ""
    test_video_path: str = ""

    yolo_model: str = "yolov8n.pt"
    yolo_imgsz: int = 960
    confidence_threshold: float = 0.35
    target_fps: float = 3.0
    preview_max_width: int = 1280

    dwell_seconds: float = 15.0
    movement_threshold: float = 0.015
    min_track_frames: int = 3
    exempt_emergency_vehicles: bool = True
    emergency_min_saturation: int = 190
    emergency_min_value: int = 225
    emergency_strobe_swing: float = 0.015
    emergency_fire_red_ratio: float = 0.35
    annotation_scale: float = 1.0
    webhook_url: str = ""

    host: str = "0.0.0.0"
    port: int = 8000
    data_dir: str = "data"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir)

    @property
    def snapshots_path(self) -> Path:
        return self.data_path / "snapshots"

    @property
    def db_path(self) -> Path:
        return self.data_path / "tracker.db"


class RuntimeConfig(BaseModel):
    """UI-editable runtime configuration persisted in SQLite."""

    nvr_host: str = "192.168.68.51"
    nvr_port: int = Field(default=554, ge=1, le=65535)
    nvr_channel: int = Field(default=7, ge=1, le=64)
    nvr_stream_type: Literal["main", "sub"] = "main"
    stream_mode: Literal["auto", "rtmp", "http_snap", "rtsp"] = "auto"
    nvr_user: str = "admin"
    nvr_pass: str = ""
    test_video_path: str = ""

    yolo_imgsz: int = Field(default=960, ge=320, le=1920)
    confidence_threshold: float = Field(default=0.35, ge=0.05, le=0.95)
    target_fps: float = Field(default=3.0, ge=0.5, le=30.0)
    preview_max_width: int = Field(default=1280, ge=320, le=3840)

    dwell_seconds: float = Field(default=15.0, ge=1.0, le=3600.0)
    movement_threshold: float = Field(default=0.015, ge=0.001, le=0.2)
    min_track_frames: int = Field(default=3, ge=1, le=60)
    exempt_emergency_vehicles: bool = True
    emergency_min_saturation: int = Field(default=190, ge=100, le=255)
    emergency_min_value: int = Field(default=225, ge=120, le=255)
    emergency_strobe_swing: float = Field(default=0.015, ge=0.002, le=0.15)
    emergency_fire_red_ratio: float = Field(default=0.35, ge=0.10, le=0.80)
    annotation_scale: float = Field(default=1.0, ge=0.4, le=3.0)
    webhook_url: str = ""
    heatmap_enabled: bool = False

    @classmethod
    def from_settings(cls, s: Settings) -> RuntimeConfig:
        return cls(
            nvr_host=s.nvr_host,
            nvr_port=s.nvr_port,
            nvr_channel=s.nvr_channel,
            nvr_stream_type=s.nvr_stream_type,
            stream_mode=s.stream_mode,
            nvr_user=s.nvr_user,
            nvr_pass=s.nvr_pass,
            test_video_path=s.test_video_path,
            yolo_imgsz=s.yolo_imgsz,
            confidence_threshold=s.confidence_threshold,
            target_fps=s.target_fps,
            preview_max_width=s.preview_max_width,
            dwell_seconds=s.dwell_seconds,
            movement_threshold=s.movement_threshold,
            min_track_frames=s.min_track_frames,
            exempt_emergency_vehicles=s.exempt_emergency_vehicles,
            emergency_min_saturation=s.emergency_min_saturation,
            emergency_min_value=s.emergency_min_value,
            emergency_strobe_swing=s.emergency_strobe_swing,
            emergency_fire_red_ratio=s.emergency_fire_red_ratio,
            annotation_scale=s.annotation_scale,
            webhook_url=s.webhook_url,
        )


def build_rtsp_url(
    host: str,
    port: int,
    channel: int,
    stream_type: Literal["main", "sub"],
    user: str,
    password: str,
) -> str:
    """Build a Reolink NVR RTSP URL with properly escaped credentials."""
    safe_user = quote(user, safe="")
    safe_pass = quote(password, safe="")
    auth = f"{safe_user}:{safe_pass}@" if safe_user or safe_pass else ""
    return f"rtsp://{auth}{host}:{port}/h264Preview_{channel:02d}_{stream_type}"


def build_rtmp_url(
    host: str,
    channel: int,
    stream_type: Literal["main", "sub"],
    user: str,
    password: str,
    port: int = 1935,
) -> str:
    """Build a Reolink NVR RTMP URL with properly escaped credentials."""
    ch_idx = max(0, channel - 1)
    stream_idx = 0 if stream_type == "main" else 1
    safe_user = quote(user, safe="")
    safe_pass = quote(password, safe="")
    return (
        f"rtmp://{host}:{port}/bcs/channel{ch_idx}_{stream_type}.bcs"
        f"?channel={ch_idx}&stream={stream_idx}&user={safe_user}&password={safe_pass}"
    )


def mask_rtsp_url(url: str) -> str:
    """Mask password in an RTSP or RTMP URL for safe logging and UI display."""
    if "password=" in url:
        prefix, rest = url.split("password=", 1)
        if "&" in rest:
            _, suffix = rest.split("&", 1)
            url = f"{prefix}password=***&{suffix}"
        else:
            url = f"{prefix}password=***"
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    creds, host_part = rest.split("@", 1)
    if ":" in creds:
        user, _ = creds.split(":", 1)
        return f"{scheme}://{user}:***@{host_part}"
    return url
