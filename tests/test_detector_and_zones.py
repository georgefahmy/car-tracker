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

        # Successful daily report automatically resets counts
        from unittest.mock import MagicMock, patch

        mock_resp = MagicMock(status_code=204)
        with patch("requests.post", return_value=mock_resp):
            ok_r = client.post("/api/report/daily/send", json={"webhook_url": "https://discord.com/api/webhooks/test/tok"})
        assert ok_r.status_code == 200
        assert ok_r.json()["ok"] is True
        assert ok_r.json()["reset"] is True
        after = db.get_stats_state()
        assert after["total_cars"] == 0
        assert after["total_people"] == 0


def test_ground_calibration_and_birdseye(tmp_path: Path) -> None:
    from app.database import GroundCalibrationModel
    from app.detector import render_birdseye_jpeg

    settings = Settings(data_dir=str(tmp_path / "data"))
    app = create_app(settings=settings, start_worker=False)
    with TestClient(app) as client:
        assert client.get("/api/calibration").json()["calibration"] is None

        calib = {
            "points": [[0.3, 0.4], [0.6, 0.4], [0.7, 0.8], [0.2, 0.8]],
            "width_ft": 20.0,
            "length_ft": 10.0,
            "grid_step_ft": 5.0,
            "show_grid": True,
        }
        put_r = client.put("/api/calibration", json=calib)
        assert put_r.status_code == 200
        body = put_r.json()
        assert body["calibration"]["width_ft"] == 20.0
        assert len(body["ground_to_img_norm"]) == 3

        proj_r = client.post("/api/calibration/project", json={"points": [[0.5, 0.5]]})
        assert proj_r.status_code == 200
        assert len(proj_r.json()["points"]) == 1

        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        jpg = render_birdseye_jpeg(frame, GroundCalibrationModel(**calib), [], [])
        assert jpg is not None and jpg[:2] == b"\xff\xd8"

        assert client.delete("/api/calibration").status_code == 200
        assert client.get("/api/calibration").json()["calibration"] is None


def test_detect_red_curb_zones() -> None:
    import cv2
    from app.database import GroundCalibrationModel
    from app.detector import detect_red_curb_zones

    # Blank gray street -> no red curb detected
    gray_frame = np.full((360, 640, 3), 120, dtype=np.uint8)
    empty_res = detect_red_curb_zones(gray_frame)
    assert empty_res["detected"] is False
    assert empty_res["zones"] == []

    # Draw a painted red curb strip along the lower-left/bottom edge
    curb_frame = gray_frame.copy()
    cv2.line(curb_frame, (120, 260), (500, 240), (20, 20, 210), thickness=12)

    res_2d = detect_red_curb_zones(curb_frame, calib=None)
    assert res_2d["detected"] is True
    assert res_2d["used_ground_calibration"] is False
    assert len(res_2d["zones"]) == 1
    assert len(res_2d["zones"][0]["points"]) == 4

    calib = GroundCalibrationModel(
        points=[[0.3, 0.45], [0.7, 0.45], [0.85, 0.85], [0.15, 0.85]],
        width_ft=20.0,
        length_ft=15.0,
    )
    res_3d = detect_red_curb_zones(curb_frame, calib=calib)
    assert res_3d["detected"] is True
    assert res_3d["used_ground_calibration"] is True
    assert len(res_3d["zones"]) == 1
    assert len(res_3d["zones"][0]["points"]) == 4


def test_two_line_speed_estimation() -> None:
    from app.database import GroundCalibrationModel

    # Calibrate 20 ft wide rectangle aligned with normalized [0.0..1.0] -> 20 ft
    calib = GroundCalibrationModel(
        points=[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
        width_ft=20.0,
        length_ft=20.0,
    )
    lm = LineCrossingMonitor()
    ln1 = CountingLineModel(id="ln1", name="Line A", p1=[0.25, 0.0], p2=[0.25, 1.0], enabled=True)
    ln2 = CountingLineModel(id="ln2", name="Line B", p1=[0.75, 0.0], p2=[0.75, 1.0], enabled=True)

    # Car moves from x=0.1 to x=0.4 across ln1 (x=0.25) between t=0.0 and t=1.0 -> t_cross_1 = 0.5s
    lm.process_tracks([_make_car(7, 0.1, 0.5)], 1000, 1000, [ln1, ln2], now_ts=0.0, calib=calib)
    lm.process_tracks([_make_car(7, 0.4, 0.5)], 1000, 1000, [ln1, ln2], now_ts=1.0, calib=calib)
    # Car moves from x=0.4 to x=0.9 across ln2 (x=0.75) between t=1.0 and t=2.0 -> t_cross_2 = 1.7s
    # Distance = 0.5 * 20 ft = 10 ft, dt = 1.2s -> 10 / 1.2 * 0.681818 = 5.68 -> 5.7 mph
    lm.process_tracks([_make_car(7, 0.9, 0.5)], 1000, 1000, [ln1, ln2], now_ts=2.0, calib=calib)

    speeds = lm.get_track_speeds()
    assert 7 in speeds
    assert abs(speeds[7] - 5.7) <= 0.2
    summary = lm.get_speed_summary()
    assert summary["calibrated"] is True
    assert summary["samples_count"] == 1
    assert summary["last_speed_mph"] == speeds[7]
    new_events = lm.consume_new_speeds()
    assert len(new_events) == 1
    assert new_events[0]["track_id"] == 7
    assert lm.consume_new_speeds() == []

    # Holding the vehicle stationary for >= 3s clears its live speed badge while keeping summary history
    lm.process_tracks([_make_car(7, 0.9, 0.5)], 1000, 1000, [ln1, ln2], now_ts=5.5, calib=calib)
    assert 7 not in lm.get_track_speeds()
    assert lm.get_speed_summary()["samples_count"] == 1


def test_alpr_plate_normalization_and_reader() -> None:
    from unittest.mock import MagicMock

    from app.alpr import LicensePlateReader, normalize_plate_candidate

    assert normalize_plate_candidate("7ABC123") == "7ABC123"
    assert normalize_plate_candidate("CA 8XYZ-991") == "8XYZ991"
    assert normalize_plate_candidate("12345A1") == "12345A1"
    assert normalize_plate_candidate("TOYOTA") is None
    assert normalize_plate_candidate("REOLINK") is None
    assert normalize_plate_candidate("2026") is None
    assert normalize_plate_candidate("AB") is None

    reader = LicensePlateReader()
    reader._ocr = MagicMock(
        return_value=(
            [
                [[[10, 10], [90, 10], [90, 30], [10, 30]], "TESLA", 0.99],
                [[[40, 150], [140, 150], [140, 180], [40, 180]], "7XYZ-421", 0.91],
            ],
            [0.02],
        )
    )
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    plate, conf = reader.read_plate(frame, bbox_xyxy=(50, 40, 250, 260))
    assert plate == "7XYZ421"
    assert abs(conf - 0.91) < 1e-3

    # Errors in OCR never raise out of read_plate
    reader._ocr = MagicMock(side_effect=RuntimeError("mock failure"))
    assert reader.read_plate(frame) == ("", 0.0)


def test_database_license_plate_migration_and_csv(tmp_path: Path) -> None:
    import sqlite3

    from app.database import Database

    db_file = tmp_path / "legacy.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE violations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            track_id INTEGER NOT NULL,
            zone_id TEXT NOT NULL,
            zone_name TEXT NOT NULL,
            vehicle_class TEXT NOT NULL,
            started_at TEXT NOT NULL,
            flagged_at TEXT NOT NULL,
            dwell_seconds REAL NOT NULL,
            snapshot_filename TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    # Initializing Database should migrate violations table cleanly without losing compatibility
    migrated_db = Database(db_file)
    rec = migrated_db.add_violation(
        track_id=5,
        zone_id="z1",
        zone_name="Red Curb",
        vehicle_class="car",
        started_at="2026-11-14T20:00:00Z",
        flagged_at="2026-11-14T20:00:20Z",
        dwell_seconds=20.0,
        snapshot_filename="violation_test.jpg",
        license_plate="7ABC123",
    )
    assert rec.license_plate == "7ABC123"
    migrated_db.close()

    settings = Settings(data_dir=str(tmp_path))
    # Point test app directly to a fresh DB inside tmp_path via db_path=data_dir/tracker.db
    app = create_app(settings=settings, start_worker=False)
    with TestClient(app) as client:
        app.state.db.add_violation(
            track_id=9,
            zone_id="z1",
            zone_name="Red Curb",
            vehicle_class="car",
            started_at="2026-11-14T20:05:00Z",
            flagged_at="2026-11-14T20:05:25Z",
            dwell_seconds=25.0,
            snapshot_filename="snap_plate.jpg",
            license_plate="7ABC123",
        )
        items = client.get("/api/violations").json()
        assert len(items) == 1
        assert items[0]["license_plate"] == "7ABC123"

        csv_r = client.get("/api/violations/export.csv")
        assert csv_r.status_code == 200
        assert "license_plate" in csv_r.text
        assert "7ABC123" in csv_r.text


def test_home_assistant_publisher_and_api(tmp_path: Path, monkeypatch) -> None:
    from unittest.mock import MagicMock, patch

    import requests

    from app.ha_integration import HomeAssistantPublisher

    monkeypatch.setenv("SUPERVISOR_TOKEN", "sup_secret_tok")
    posted: list[tuple[str, str, dict]] = []

    def _fake_post(url: str, headers: dict | None = None, json: dict | None = None, timeout: int = 4):
        posted.append((url, (headers or {}).get("Authorization", ""), json or {}))
        return MagicMock(status_code=200, text="{}")

    settings = Settings(data_dir=str(tmp_path / "data"), nvr_pass="pw", ha_token="custom_secret")
    app = create_app(settings=settings, start_worker=False)
    with TestClient(app) as client:
        # Config masks ha_token and exposes ha_supervisor_detected
        cfg = client.get("/api/config").json()
        assert cfg["ha_token"] == "***"
        assert cfg["has_ha_token"] is True
        assert cfg["ha_supervisor_detected"] is True

        # Updating config with "***" preserves existing ha_token
        put_r = client.put("/api/config", json=cfg)
        assert put_r.status_code == 200
        assert app.state.worker.config.ha_token == "custom_secret"

        with patch("requests.post", side_effect=_fake_post):
            ha_r = client.post("/api/ha/test", json={"ha_url": "", "ha_token": ""})
        assert ha_r.status_code == 200
        body = ha_r.json()
        assert body["ok"] is True
        assert body["mode"] == "supervisor"
        assert body["entities"] == 8
        urls = [u for u, _, _ in posted]
        assert any(u == "http://supervisor/core/api/states/sensor.car_tracker_vehicles_today" for u in urls)
        assert any(u == "http://supervisor/core/api/states/sensor.car_tracker_last_plate" for u in urls)
        assert any(u == "http://supervisor/core/api/states/binary_sensor.car_tracker_red_zone_occupied" for u in urls)
        assert all(auth == "Bearer sup_secret_tok" for _, auth, _ in posted)

    # Verify fallback to 172.30.32.2 when `supervisor` hostname fails DNS resolution
    pub = HomeAssistantPublisher()
    fallback_urls: list[str] = []

    def _fake_fallback_post(url: str, headers: dict | None = None, json: dict | None = None, timeout: int = 4):
        fallback_urls.append(url)
        if url.startswith("http://supervisor"):
            raise requests.ConnectionError("DNS resolution failed")
        return MagicMock(status_code=200, text="{}")

    with patch("requests.post", side_effect=_fake_fallback_post):
        res = pub.publish_states({"vehicles_today": 3, "people_today": 1})
    assert res["ok"] is True
    assert any(u.startswith("http://172.30.32.2/core/api/states/") for u in fallback_urls)


def test_onnx_and_openvino_gpu_config_and_fallback() -> None:
    import sys
    import types

    from app.config import RuntimeConfig
    from app.detector import ObjectDetector

    cfg_onnx = RuntimeConfig(inference_backend="onnx")
    assert cfg_onnx.inference_backend == "onnx"
    cfg_igpu = RuntimeConfig(inference_backend="openvino_gpu")
    assert cfg_igpu.inference_backend == "openvino_gpu"

    fake_ov = types.ModuleType("openvino")

    class _FakeCore:
        available_devices = ["CPU"]

    fake_ov.Core = _FakeCore  # type: ignore[attr-defined]
    prev = sys.modules.get("openvino")
    sys.modules["openvino"] = fake_ov
    try:
        res = ObjectDetector._try_load_openvino(
            yolo_cls=object,
            model_name="yolov8n.pt",
            imgsz=640,
            models_dir=None,
            require_gpu=True,
        )
        assert res is None
    finally:
        if prev is None:
            sys.modules.pop("openvino", None)
        else:
            sys.modules["openvino"] = prev


def test_track_stabilizer_static_id_and_class_stability() -> None:
    from app.detector import TrackStabilizer

    stabilizer = TrackStabilizer()
    w, h = 1000, 1000

    def _det(
        tid: int,
        cls_id: int = 2,
        cls_name: str = "car",
        conf: float = 0.88,
        dx: float = 0.0,
        dy: float = 0.0,
    ) -> TrackedDetection:
        return TrackedDetection(
            track_id=tid,
            class_id=cls_id,
            class_name=cls_name,
            confidence=conf,
            bbox_xyxy=(400.0 + dx, 700.0 + dy, 540.0 + dx, 840.0 + dy),
        )

    # First 5 frames build stationary consensus with raw_id=1
    for i in range(5):
        out = stabilizer.stabilize([_det(1, conf=0.88, dx=float(i % 2))], w, h, now_ts=10.0 + i * 0.5, conf_threshold=0.35)
        assert len(out) == 1
        assert out[0].track_id == 1
        assert out[0].class_name == "car"

    # Frame with overlapping duplicate (car + truck) and class flip on raw_id=9 -> keeps track_id=1, class="car"
    out_dup = stabilizer.stabilize(
        [
            _det(9, cls_id=7, cls_name="truck", conf=0.72, dx=2.0),
            _det(10, cls_id=2, cls_name="car", conf=0.55, dx=3.0),
        ],
        w,
        h,
        now_ts=13.0,
        conf_threshold=0.35,
    )
    assert len(out_dup) == 1
    assert out_dup[0].track_id == 1
    assert out_dup[0].class_name == "car"

    # Confidence dip on raw_id=14 still stitches to canonical track 1
    out_dip = stabilizer.stabilize(
        [_det(14, cls_id=2, cls_name="car", conf=0.22, dx=1.0)],
        w,
        h,
        now_ts=13.5,
        conf_threshold=0.35,
    )
    assert len(out_dip) == 1
    assert out_dip[0].track_id == 1

    # 1-frame dropout (within 2.0s) coasts stationary car #1 without blinking out
    out_coast = stabilizer.stabilize([], w, h, now_ts=14.0, conf_threshold=0.35)
    assert len(out_coast) == 1
    assert out_coast[0].track_id == 1

    # After >180s absence, expired stationary track is pruned and a new vehicle gets canonical track_id=2
    out_new = stabilizer.stabilize(
        [_det(99, cls_id=2, cls_name="car", conf=0.90)],
        w,
        h,
        now_ts=200.0,
        conf_threshold=0.35,
    )
    assert len(out_new) == 1
    assert out_new[0].track_id == 2


def test_red_zone_monitor_dwell_grace_and_static_dedup() -> None:
    monitor = RedZoneMonitor()
    zone = ZoneModel(
        id="z1",
        name="Curb Red Zone",
        points=[[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]],
        enabled=True,
    )

    # Frame 1 & 2 count car #1 inside red zone
    r1 = monitor.process_tracks(
        [_make_car(1, 0.5, 0.5)], 1000, 1000, [zone], now_ts=100.0, dwell_seconds=10.0, movement_threshold=0.015, min_track_frames=2
    )
    r2 = monitor.process_tracks(
        [_make_car(1, 0.5, 0.5)], 1000, 1000, [zone], now_ts=104.0, dwell_seconds=10.0, movement_threshold=0.015, min_track_frames=2
    )
    assert (r1.new_unique_cars + r2.new_unique_cars) == 1
    assert r2.track_statuses[1]["dwell_seconds"] == 4.0

    # Brief 1-frame dropout at t=106s (2s gap <= 5s dwell grace) preserves dwell state
    r3 = monitor.process_tracks(
        [], 1000, 1000, [zone], now_ts=106.0, dwell_seconds=10.0, movement_threshold=0.015, min_track_frames=2
    )
    assert r3.active_cars == 0

    # Car #1 reappears at t=111s -> total stationary elapsed 11s >= 10s -> triggers violation
    r4 = monitor.process_tracks(
        [_make_car(1, 0.501, 0.501)], 1000, 1000, [zone], now_ts=111.0, dwell_seconds=10.0, movement_threshold=0.015, min_track_frames=2
    )
    assert len(r4.new_violations) == 1

    # Stationary car re-appearing with a brand-new ID (#77) at same parked location within 300s must not inflate unique_cars
    r5 = monitor.process_tracks(
        [_make_car(77, 0.502, 0.502)], 1000, 1000, [zone], now_ts=120.0, dwell_seconds=10.0, movement_threshold=0.015, min_track_frames=2
    )
    r6 = monitor.process_tracks(
        [_make_car(77, 0.502, 0.502)], 1000, 1000, [zone], now_ts=121.0, dwell_seconds=10.0, movement_threshold=0.015, min_track_frames=2
    )
    assert r5.new_unique_cars == 0
    assert r6.new_unique_cars == 0
    assert r6.new_total_cars == 0


def test_line_crossing_ignores_stationary_jitter() -> None:
    lm = LineCrossingMonitor()
    line = CountingLineModel(id="ln1", name="Curb Line", p1=[0.5, 0.2], p2=[0.5, 0.8], enabled=True)

    # Parked car jittering +-0.005 back and forth across x=0.5 over 20 frames -> 0 crossings
    for i in range(20):
        cx = 0.495 if i % 2 == 0 else 0.505
        inc = lm.process_tracks([_make_car(1, cx, 0.5)], 1000, 1000, [line], now_ts=float(i + 1))
        assert inc == {}

    # Car spawning right near the line (0.495 -> 0.505 -> 0.535) only counts once committed across deadband + travel
    inc_a = lm.process_tracks([_make_car(2, 0.495, 0.5)], 1000, 1000, [line], now_ts=30.0)
    inc_b = lm.process_tracks([_make_car(2, 0.505, 0.5)], 1000, 1000, [line], now_ts=31.0)
    inc_c = lm.process_tracks([_make_car(2, 0.535, 0.5)], 1000, 1000, [line], now_ts=32.0)
    assert inc_a == {}
    assert inc_b == {}
    assert inc_c == {"ln1": (0, 1)}

    # Normal moving car (0.3 -> 0.7) counts immediately in one step
    lm.process_tracks([_make_car(3, 0.3, 0.5)], 1000, 1000, [line], now_ts=40.0)
    inc_fast = lm.process_tracks([_make_car(3, 0.7, 0.5)], 1000, 1000, [line], now_ts=41.0)
    assert inc_fast == {"ln1": (0, 1)}




