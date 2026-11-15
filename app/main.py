from __future__ import annotations

import asyncio
import csv
import io
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.config import RuntimeConfig, Settings, build_rtmp_url, build_rtsp_url
from app.database import CountingLineModel, Database, GroundCalibrationModel, ViolationRecord, ZoneModel
from app.detector import compute_ground_transforms, project_points, render_birdseye_jpeg
from app.stream import StreamWorker, test_stream_connection

BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


class ConfigUpdatePayload(RuntimeConfig):
    """Payload for updating runtime configuration; '***' preserves existing password."""

    pass


class StreamTestPayload(BaseModel):
    nvr_host: str | None = None
    nvr_port: int | None = None
    nvr_channel: int | None = None
    nvr_stream_type: str | None = None
    stream_mode: str | None = None
    nvr_user: str | None = None
    nvr_pass: str | None = None
    test_video_path: str | None = None


class WebhookTestPayload(BaseModel):
    webhook_url: str | None = None


class BevProjectPayload(BaseModel):
    points: list[list[float]] = []


def create_app(settings: Settings | None = None, start_worker: bool = True) -> FastAPI:
    cfg_settings = settings or Settings()
    cfg_settings.snapshots_path.mkdir(parents=True, exist_ok=True)

    db = Database(cfg_settings.db_path)
    worker = StreamWorker(cfg_settings, db)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncGenerator[None, None]:
        if start_worker:
            worker.start()
        try:
            yield
        finally:
            worker.stop()
            db.close()

    app = FastAPI(title="Reolink Car & Red-Zone Tracker", lifespan=lifespan)
    app.state.db = db
    app.state.worker = worker
    app.state.settings = cfg_settings

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
    app.mount("/snapshots", StaticFiles(directory=str(cfg_settings.snapshots_path)), name="snapshots")

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/stream/frame.jpg")
    async def video_frame() -> Response:
        jpg = worker.get_latest_jpeg()
        if not jpg:
            raise HTTPException(status_code=503, detail="No frame available yet")
        return Response(
            content=jpg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/api/stream/video")
    async def video_stream() -> StreamingResponse:
        async def frame_generator() -> AsyncGenerator[bytes, None]:
            while True:
                jpg = worker.get_latest_jpeg()
                if jpg:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n\r\n" + jpg + b"\r\n"
                    )
                await asyncio.sleep(0.03)

        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/stats")
    async def get_stats() -> dict:
        state = worker.get_live_state()
        state["hourly_traffic"] = db.get_hourly_traffic(limit=24)
        return state

    @app.post("/api/stats/reset")
    async def reset_stats() -> dict:
        db.reset_stats()
        db.reset_line_counts()
        worker.monitor.reset()
        worker.line_monitor.reset()
        return {"ok": True}

    @app.get("/api/zones", response_model=list[ZoneModel])
    async def get_zones() -> list[ZoneModel]:
        return db.get_zones()

    @app.put("/api/zones", response_model=list[ZoneModel])
    async def put_zones(zones: list[ZoneModel]) -> list[ZoneModel]:
        return db.replace_zones(zones)

    @app.get("/api/lines", response_model=list[CountingLineModel])
    async def get_lines() -> list[CountingLineModel]:
        return db.get_counting_lines()

    @app.put("/api/lines", response_model=list[CountingLineModel])
    async def put_lines(lines: list[CountingLineModel]) -> list[CountingLineModel]:
        return db.replace_counting_lines(lines)

    @app.get("/api/violations", response_model=list[ViolationRecord])
    async def get_violations(
        limit: int = Query(default=50, ge=1, le=200),
        offset: int = Query(default=0, ge=0),
    ) -> list[ViolationRecord]:
        return db.get_violations(limit=limit, offset=offset)

    @app.get("/api/violations/export.csv")
    async def export_violations_csv() -> Response:
        rows = db.get_violations(limit=5000, offset=0)
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["id", "flagged_at", "started_at", "dwell_seconds", "zone_name", "vehicle_class", "track_id", "snapshot_filename"])
        for r in rows:
            writer.writerow([r.id, r.flagged_at, r.started_at, r.dwell_seconds, r.zone_name, r.vehicle_class, r.track_id, r.snapshot_filename])
        return Response(
            content=buf.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=violations.csv"},
        )

    @app.delete("/api/violations/{violation_id}")
    async def delete_violation(violation_id: int) -> dict:
        fname = db.delete_violation(violation_id)
        if fname is None:
            raise HTTPException(status_code=404, detail="Violation not found")
        snap = cfg_settings.snapshots_path / fname
        if snap.exists():
            snap.unlink()
        return {"ok": True}

    @app.get("/api/config")
    async def get_config() -> dict:
        data = worker.config.model_dump()
        has_pass = bool(data.get("nvr_pass"))
        data["nvr_pass"] = "***" if has_pass else ""
        data["has_password"] = has_pass
        return data

    @app.put("/api/config")
    async def put_config(payload: ConfigUpdatePayload) -> dict:
        if payload.nvr_pass == "***":
            payload.nvr_pass = worker.config.nvr_pass
        worker.update_config(RuntimeConfig(**payload.model_dump()))
        data = worker.config.model_dump()
        has_pass = bool(data.get("nvr_pass"))
        data["nvr_pass"] = "***" if has_pass else ""
        data["has_password"] = has_pass
        return data

    @app.post("/api/stream/test")
    async def test_stream(payload: StreamTestPayload) -> dict:
        cur = worker.config
        test_vid = payload.test_video_path if payload.test_video_path is not None else cur.test_video_path
        if test_vid.strip():
            source = test_vid.strip()
        else:
            raw_pass = payload.nvr_pass
            if raw_pass in (None, "***"):
                raw_pass = cur.nvr_pass
            stream_type = payload.nvr_stream_type or cur.nvr_stream_type
            if stream_type not in ("main", "sub"):
                stream_type = "main"
            stream_mode = payload.stream_mode or cur.stream_mode
            if stream_mode == "rtmp":
                source = build_rtmp_url(
                    host=payload.nvr_host or cur.nvr_host,
                    channel=payload.nvr_channel or cur.nvr_channel,
                    stream_type=stream_type,  # type: ignore[arg-type]
                    user=payload.nvr_user if payload.nvr_user is not None else cur.nvr_user,
                    password=raw_pass,
                )
            else:
                source = build_rtsp_url(
                    host=payload.nvr_host or cur.nvr_host,
                    port=payload.nvr_port or cur.nvr_port,
                    channel=payload.nvr_channel or cur.nvr_channel,
                    stream_type=stream_type,  # type: ignore[arg-type]
                    user=payload.nvr_user if payload.nvr_user is not None else cur.nvr_user,
                    password=raw_pass,
                )
        return await asyncio.to_thread(test_stream_connection, source)

    @app.post("/api/webhook/test")
    async def test_webhook(payload: WebhookTestPayload) -> dict:
        url = (payload.webhook_url if payload.webhook_url is not None else worker.config.webhook_url).strip()
        if not url:
            return {"ok": False, "error": "Enter a webhook URL first."}

        def _send_test() -> dict:
            import requests

            msg = "🚨 Car Tracker Test Alert: Your red-zone violation webhook is configured and working!"
            try:
                if "discord.com/api/webhooks" in url:
                    r = requests.post(url, json={"content": msg}, timeout=8)
                else:
                    r = requests.post(url, json={"text": msg, "content": msg, "test": True}, timeout=8)
                if 200 <= r.status_code < 300:
                    return {"ok": True, "status_code": r.status_code}
                return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:160]}"}
            except Exception as exc:
                return {"ok": False, "error": str(exc)}

        return await asyncio.to_thread(_send_test)

    @app.post("/api/report/daily/send")
    async def send_daily_report_now(payload: WebhookTestPayload | None = None) -> dict:
        url = payload.webhook_url if payload and payload.webhook_url is not None else None
        return await asyncio.to_thread(worker.send_daily_report, url)

    @app.post("/api/heatmap/toggle")
    async def toggle_heatmap(enabled: bool | None = None) -> dict:
        new_state = worker.toggle_heatmap(enabled)
        return {"heatmap_enabled": new_state}

    @app.post("/api/heatmap/reset")
    async def reset_heatmap() -> dict:
        worker.reset_heatmap()
        return {"ok": True}

    def _calibration_response(calib: GroundCalibrationModel | None) -> dict:
        if calib is None:
            return {"calibration": None, "ground_to_img_norm": None, "bev_norm_to_img_norm": None}
        tf = compute_ground_transforms(calib)
        return {
            "calibration": calib.model_dump(),
            "ground_to_img_norm": tf["ground_to_img_norm"].tolist(),
            "bev_norm_to_img_norm": tf["bev_norm_to_img_norm"].tolist(),
        }

    @app.get("/api/calibration")
    async def get_calibration() -> dict:
        return _calibration_response(db.get_ground_calibration())

    @app.put("/api/calibration")
    async def put_calibration(payload: GroundCalibrationModel) -> dict:
        saved = db.save_ground_calibration(payload)
        return _calibration_response(saved)

    @app.delete("/api/calibration")
    async def delete_calibration() -> dict:
        db.delete_ground_calibration()
        return {"ok": True}

    @app.get("/api/calibration/birdseye.jpg")
    async def get_birdseye_image() -> Response:
        calib = db.get_ground_calibration()
        if calib is None:
            raise HTTPException(status_code=404, detail="Ground calibration not configured")
        frame = worker.get_latest_raw_frame()
        if frame is None:
            raise HTTPException(status_code=503, detail="No camera frame available yet")
        jpg = await asyncio.to_thread(
            render_birdseye_jpeg, frame, calib, db.get_zones(), db.get_counting_lines()
        )
        if not jpg:
            raise HTTPException(status_code=500, detail="Failed to render bird's-eye view")
        return Response(
            content=jpg,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.post("/api/calibration/project")
    async def project_bev_points(payload: BevProjectPayload) -> dict:
        calib = db.get_ground_calibration()
        if calib is None:
            raise HTTPException(status_code=404, detail="Ground calibration not configured")
        tf = compute_ground_transforms(calib)
        cam_pts = project_points(payload.points, tf["bev_norm_to_img_norm"])
        return {"points": cam_pts}

    return app


app = create_app()
