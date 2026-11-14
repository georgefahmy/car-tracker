from __future__ import annotations

from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings, build_rtsp_url, mask_rtsp_url
from app.database import CountingLineModel, ZoneModel
from app.detector import (
    PERSON_CLASS_ID,
    EmergencyVehicleClassifier,
    HotspotTracker,
    LineCrossingMonitor,
    RedZoneMonitor,
    TrackedDetection,
    filter_excluded_tracks,
    point_in_normalized_polygon,
)
from app.main import create_app


def _make_car(track_id: int, cx: float = 0.5, y2: float = 0.5, w: int = 1000, h: int = 1000) -> TrackedDetection:
    bw = 0.1 * w
    bh = 0.1 * h
    x1 = cx * w - bw / 2
    x2 = cx * w + bw / 2
    y1 = y2 * h - bh
    return TrackedDetection(
        track_id=track_id,
        class_id=2,
        class_name="car",
        confidence=0.9,
        bbox_xyxy=(x1, y1, x2, y2 * h),
    )


def _make_person(track_id: int, cx: float = 0.5, y2: float = 0.5, w: int = 1000, h: int = 1000) -> TrackedDetection:
    bw = 0.04 * w
    bh = 0.12 * h
    x1 = cx * w - bw / 2
    x2 = cx * w + bw / 2
    y1 = y2 * h - bh
    return TrackedDetection(
        track_id=track_id,
        class_id=PERSON_CLASS_ID,
        class_name="person",
        confidence=0.85,
        bbox_xyxy=(x1, y1, x2, y2 * h),
    )


def test_rtsp_url_builder_and_masking() -> None:
    url = build_rtsp_url(
        host="192.168.68.51",
        port=554,
        channel=7,
        stream_type="main",
        user="admin",
        password="p@ss:word/1!",
    )
    assert url == "rtsp://admin:p%40ss%3Aword%2F1%21@192.168.68.51:554/h264Preview_07_main"
    assert mask_rtsp_url(url) == "rtsp://admin:***@192.168.68.51:554/h264Preview_07_main"


def test_point_in_normalized_polygon() -> None:
    square = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
    assert point_in_normalized_polygon(0.5, 0.5, square) is True
    assert point_in_normalized_polygon(0.1, 0.1, square) is False


def test_monitor_counts_and_red_zone_violation() -> None:
    monitor = RedZoneMonitor()
    zone = ZoneModel(
        id="z1",
        name="Curb Red Zone",
        points=[[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
        enabled=True,
    )

    # Frame 1 & 2: car #1 and person #2 appear (min_track_frames=2)
    r1 = monitor.process_tracks(
        [_make_car(1, 0.5, 0.5), _make_person(2, 0.5, 0.5)],
        1000,
        1000,
        [zone],
        now_ts=100.0,
        dwell_seconds=10.0,
        movement_threshold=0.015,
        min_track_frames=2,
    )
    assert r1.active_cars == 1
    assert r1.active_people == 1
    assert r1.new_unique_cars == 0  # not yet at min_track_frames=2

    r2 = monitor.process_tracks(
        [_make_car(1, 0.5, 0.5), _make_person(2, 0.5, 0.5)],
        1000,
        1000,
        [zone],
        now_ts=105.0,
        dwell_seconds=10.0,
        movement_threshold=0.015,
        min_track_frames=2,
    )
    assert r2.new_unique_cars == 1
    assert r2.new_total_cars == 1
    assert r2.new_unique_people == 1
    assert r2.new_total_people == 1
    assert r2.new_violations == []

    # Frame 3 at t=111s: car #1 stationary for 11s >= 10s -> violation! Person #2 ignored for parking.
    r3 = monitor.process_tracks(
        [_make_car(1, 0.502, 0.501), _make_person(2, 0.5, 0.5)],
        1000,
        1000,
        [zone],
        now_ts=111.0,
        dwell_seconds=10.0,
        movement_threshold=0.015,
        min_track_frames=2,
    )
    assert len(r3.new_violations) == 1
    assert r3.new_violations[0].track_id == 1
    assert r3.new_violations[0].vehicle_class == "car"

    # Frame 4: same parked car must not trigger a duplicate violation
    r4 = monitor.process_tracks(
        [_make_car(1, 0.502, 0.501)],
        1000,
        1000,
        [zone],
        now_ts=120.0,
        dwell_seconds=10.0,
        movement_threshold=0.015,
        min_track_frames=2,
    )
    assert r4.new_violations == []


def test_moving_car_does_not_trigger_violation() -> None:
    monitor = RedZoneMonitor()
    zone = ZoneModel(
        id="z1",
        name="Curb Red Zone",
        points=[[0.1, 0.3], [0.9, 0.3], [0.9, 0.7], [0.1, 0.7]],
        enabled=True,
    )
    for step, cx in enumerate([0.2, 0.4, 0.6, 0.8]):
        res = monitor.process_tracks(
            [_make_car(10, cx, 0.5)],
            1000,
            1000,
            [zone],
            now_ts=step * 5.0,
            dwell_seconds=10.0,
            movement_threshold=0.015,
            min_track_frames=1,
        )
        assert res.new_violations == []


def test_hotspot_tracker() -> None:
    ht = HotspotTracker(grid_w=64, grid_h=36)
    ht.update([_make_car(1, 0.5, 0.5)], 1000, 1000)
    assert float(ht.grid.max()) > 0.5
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    rendered = ht.render_overlay(frame)
    assert rendered.shape == frame.shape
    ht.reset()
    assert float(ht.grid.max()) == 0.0


def test_api_endpoints(tmp_path: Path) -> None:
    settings = Settings(data_dir=str(tmp_path / "data"), nvr_pass="secret123")
    app = create_app(settings=settings, start_worker=False)

    with TestClient(app) as client:
        assert client.get("/").status_code == 200

        # Config masks password
        cfg_resp = client.get("/api/config")
        assert cfg_resp.status_code == 200
        cfg = cfg_resp.json()
        assert cfg["nvr_pass"] == "***"
        assert cfg["has_password"] is True
        assert cfg["nvr_channel"] == 7

        # Updating config with '***' preserves existing password
        cfg["target_fps"] = 2.0
        put_resp = client.put("/api/config", json=cfg)
        assert put_resp.status_code == 200
        assert app.state.worker.config.nvr_pass == "secret123"
        assert app.state.worker.config.target_fps == 2.0

        # Zones CRUD
        zones_payload = [
            {"id": "z1", "name": "Balcony Curb", "points": [[0.1, 0.5], [0.9, 0.5], [0.9, 0.9], [0.1, 0.9]], "enabled": True, "color": "#ef4444"}
        ]
        assert client.put("/api/zones", json=zones_payload).status_code == 200
        assert len(client.get("/api/zones").json()) == 1

        # Heatmap toggle
        t1 = client.post("/api/heatmap/toggle").json()
        assert t1["heatmap_enabled"] is True

        # Stats
        stats = client.get("/api/stats").json()
        assert "unique_cars" in stats
        assert "unique_people" in stats
        assert "hourly_traffic" in stats


def test_emergency_vehicle_classification_and_exemption() -> None:
    clf = EmergencyVehicleClassifier()
    frame = np.zeros((400, 600, 3), dtype=np.uint8)

    # 1. Normal grey car (no lightbar) -> not emergency
    frame[100:200, 50:150] = (120, 120, 120)
    normal_car = TrackedDetection(
        track_id=1, class_id=2, class_name="car", confidence=0.9, bbox_xyxy=(50, 100, 150, 200)
    )

    # 2. Police car with bright blue+red lightbar on roof (top 30%)
    frame[100:200, 200:300] = (40, 40, 40)
    frame[105:118, 225:250] = (255, 0, 0)  # bright blue (BGR)
    frame[105:118, 250:275] = (0, 0, 255)  # bright red (BGR)
    police_car = TrackedDetection(
        track_id=2, class_id=2, class_name="car", confidence=0.9, bbox_xyxy=(200, 100, 300, 200)
    )

    # 3. Fire truck (truck with vivid red body)
    frame[100:220, 350:500] = (0, 0, 220)  # vivid red (BGR)
    fire_truck = TrackedDetection(
        track_id=3, class_id=7, class_name="truck", confidence=0.9, bbox_xyxy=(350, 100, 500, 220)
    )

    clf.update_detections(frame, [normal_car, police_car, fire_truck])
    assert normal_car.is_emergency is False
    assert police_car.is_emergency is True
    assert fire_truck.is_emergency is True

    # Verify RedZoneMonitor exempts emergency vehicles when enabled
    monitor = RedZoneMonitor()
    zone = ZoneModel(
        id="z1",
        name="Curb Red Zone",
        points=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        enabled=True,
    )
    monitor.process_tracks(
        [police_car, fire_truck], 600, 400, [zone], now_ts=0.0, dwell_seconds=5.0, movement_threshold=0.02, min_track_frames=1
    )
    res = monitor.process_tracks(
        [police_car, fire_truck], 600, 400, [zone], now_ts=10.0, dwell_seconds=5.0, movement_threshold=0.02, min_track_frames=1
    )
    assert res.new_violations == []
    assert res.track_statuses[2]["emergency"] is True
    assert res.track_statuses[3]["emergency"] is True


def test_line_crossing_monitor_and_api(tmp_path: Path) -> None:
    lm = LineCrossingMonitor()
    line = CountingLineModel(
        id="ln1", name="Main Road", p1=[0.5, 0.2], p2=[0.5, 0.8], enabled=True
    )

    # Car 1 crosses left-to-right (0.4 -> 0.6), Car 2 crosses right-to-left (0.6 -> 0.4)
    inc1 = lm.process_tracks([_make_car(1, 0.4, 0.5), _make_car(2, 0.6, 0.5)], 1000, 1000, [line], 1.0)
    assert inc1 == {}

    inc2 = lm.process_tracks([_make_car(1, 0.6, 0.5), _make_car(2, 0.4, 0.5)], 1000, 1000, [line], 2.0)
    assert inc2 == {"ln1": (1, 1)}

    # Same cars moving further should not double-count on the same line
    inc3 = lm.process_tracks([_make_car(1, 0.7, 0.5), _make_car(2, 0.3, 0.5)], 1000, 1000, [line], 3.0)
    assert inc3 == {}

    # API CRUD for /api/lines
    settings = Settings(data_dir=str(tmp_path / "data"))
    app = create_app(settings=settings, start_worker=False)
    with TestClient(app) as client:
        put_r = client.put(
            "/api/lines",
            json=[{"id": "ln1", "name": "Main Road", "p1": [0.5, 0.2], "p2": [0.5, 0.8], "enabled": True}],
        )
        assert put_r.status_code == 200
        lines_r = client.get("/api/lines").json()
        assert len(lines_r) == 1
        assert lines_r[0]["name"] == "Main Road"

        # CSV export endpoint
        csv_r = client.get("/api/violations/export.csv")
        assert csv_r.status_code == 200
        assert "zone_name" in csv_r.text

        # Webhook test validation
        wh_r = client.post("/api/webhook/test", json={"webhook_url": ""})
        assert wh_r.status_code == 200
        assert wh_r.json()["ok"] is False


def test_exclusion_mask_filters_tracks() -> None:
    mask = ZoneModel(
        id="m1",
        name="Balcony",
        points=[[0.0, 0.7], [1.0, 0.7], [1.0, 1.0], [0.0, 1.0]],
        enabled=True,
        zone_type="exclusion",
    )
    street_car = _make_car(1, cx=0.5, y2=0.4)
    balcony_car = _make_car(2, cx=0.5, y2=0.85)
    kept = filter_excluded_tracks([street_car, balcony_car], 1000, 1000, [mask])
    assert [t.track_id for t in kept] == [1]


def test_daily_report_summary_and_endpoint(tmp_path: Path) -> None:
    from app.database import Database
    from app.stream import StreamWorker

    settings = Settings(data_dir=str(tmp_path / "data"))
    app = create_app(settings=settings, start_worker=False)
    with TestClient(app) as client:
        db: Database = app.state.db
        db.increment_stats(unique_cars=4, total_cars=7, unique_people=2, total_people=3)
        summary = db.get_daily_summary()
        assert summary["vehicles_today"] == 7
        assert summary["people_today"] == 3
        msg = StreamWorker.build_daily_report_message(summary)
        assert "Daily Car Tracker Report" in msg
        assert "**Vehicles Today:** 7" in msg

        # Empty webhook returns ok=False cleanly
        r = client.post("/api/report/daily/send", json={"webhook_url": ""})
        assert r.status_code == 200
        assert r.json()["ok"] is False


