from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ZoneModel(BaseModel):
    """Normalized polygon zone ([0..1] coordinates)."""

    id: str
    name: str
    points: list[list[float]] = Field(min_length=3)
    enabled: bool = True
    color: str = "#ef4444"
    zone_type: Literal["red_zone", "exclusion", "person_zone"] = "red_zone"


class CountingLineModel(BaseModel):
    """Normalized 2-point line ([0..1] coordinates) for traffic flow counting and/or 2-line speed."""

    id: str
    name: str
    p1: list[float] = Field(min_length=2, max_length=2)
    p2: list[float] = Field(min_length=2, max_length=2)
    enabled: bool = True
    color: str = "#38bdf8"
    count_a_to_b: int = 0
    count_b_to_a: int = 0
    line_role: Literal["both", "count", "speed"] = "both"


class GroundCalibrationModel(BaseModel):
    """4-point ground-plane homography calibration in normalized [0..1] image coords."""

    points: list[list[float]] = Field(min_length=4, max_length=4)
    width_ft: float = Field(default=20.0, gt=0.5, le=500.0)
    length_ft: float = Field(default=10.0, gt=0.5, le=500.0)
    grid_step_ft: float = Field(default=5.0, gt=0.5, le=100.0)
    show_grid: bool = True


class ViolationRecord(BaseModel):
    """Stored illegal red-zone parking or speeding violation."""

    id: int
    track_id: int
    zone_id: str
    zone_name: str
    vehicle_class: str
    started_at: str
    flagged_at: str
    dwell_seconds: float
    snapshot_filename: str
    license_plate: str = ""


class Database:
    """Thread-safe SQLite persistence for zones, violations, stats, and config."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("PRAGMA journal_mode=WAL;")
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS zones (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    points_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    color TEXT NOT NULL DEFAULT '#ef4444',
                    zone_type TEXT NOT NULL DEFAULT 'red_zone'
                );

                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER NOT NULL,
                    zone_id TEXT NOT NULL,
                    zone_name TEXT NOT NULL,
                    vehicle_class TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    flagged_at TEXT NOT NULL,
                    dwell_seconds REAL NOT NULL,
                    snapshot_filename TEXT NOT NULL,
                    license_plate TEXT NOT NULL DEFAULT ''
                );

                CREATE TABLE IF NOT EXISTS stats_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    unique_cars INTEGER NOT NULL DEFAULT 0,
                    total_cars INTEGER NOT NULL DEFAULT 0,
                    unique_people INTEGER NOT NULL DEFAULT 0,
                    total_people INTEGER NOT NULL DEFAULT 0
                );

                INSERT OR IGNORE INTO stats_state (id, unique_cars, total_cars, unique_people, total_people)
                VALUES (1, 0, 0, 0, 0);

                CREATE TABLE IF NOT EXISTS hourly_traffic (
                    hour_bucket TEXT PRIMARY KEY,
                    vehicles_count INTEGER NOT NULL DEFAULT 0,
                    people_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS app_config (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    config_json TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS meta_state (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS counting_lines (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    p1_json TEXT NOT NULL,
                    p2_json TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 1,
                    color TEXT NOT NULL DEFAULT '#38bdf8',
                    count_a_to_b INTEGER NOT NULL DEFAULT 0,
                    count_b_to_a INTEGER NOT NULL DEFAULT 0,
                    line_role TEXT NOT NULL DEFAULT 'both'
                );
                """
            )
            cur.execute("PRAGMA table_info(zones)")
            cols = {row[1] for row in cur.fetchall()}
            if "zone_type" not in cols:
                cur.execute("ALTER TABLE zones ADD COLUMN zone_type TEXT NOT NULL DEFAULT 'red_zone'")
            cur.execute("PRAGMA table_info(violations)")
            vcols = {row[1] for row in cur.fetchall()}
            if "license_plate" not in vcols:
                cur.execute("ALTER TABLE violations ADD COLUMN license_plate TEXT NOT NULL DEFAULT ''")
            cur.execute("PRAGMA table_info(stats_state)")
            scols = {row[1] for row in cur.fetchall()}
            if "person_alerts" not in scols:
                cur.execute("ALTER TABLE stats_state ADD COLUMN person_alerts INTEGER NOT NULL DEFAULT 0")
            cur.execute("PRAGMA table_info(counting_lines)")
            lcols = {row[1] for row in cur.fetchall()}
            if "line_role" not in lcols:
                cur.execute("ALTER TABLE counting_lines ADD COLUMN line_role TEXT NOT NULL DEFAULT 'both'")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- Zones ---

    def get_zones(self) -> list[ZoneModel]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, points_json, enabled, color, zone_type FROM zones"
            ).fetchall()
        return [
            ZoneModel(
                id=r["id"],
                name=r["name"],
                points=json.loads(r["points_json"]),
                enabled=bool(r["enabled"]),
                color=r["color"],
                zone_type=r["zone_type"] if r["zone_type"] in ("red_zone", "exclusion", "person_zone") else "red_zone",
            )
            for r in rows
        ]

    def replace_zones(self, zones: list[ZoneModel]) -> list[ZoneModel]:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("DELETE FROM zones")
            for z in zones:
                cur.execute(
                    "INSERT INTO zones (id, name, points_json, enabled, color, zone_type) VALUES (?, ?, ?, ?, ?, ?)",
                    (z.id, z.name, json.dumps(z.points), int(z.enabled), z.color, z.zone_type),
                )
            self._conn.commit()
        return zones

    # --- Counting Lines ---

    def get_counting_lines(self) -> list[CountingLineModel]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, p1_json, p2_json, enabled, color, count_a_to_b, count_b_to_a, line_role FROM counting_lines"
            ).fetchall()
        return [
            CountingLineModel(
                id=r["id"],
                name=r["name"],
                p1=json.loads(r["p1_json"]),
                p2=json.loads(r["p2_json"]),
                enabled=bool(r["enabled"]),
                color=r["color"],
                count_a_to_b=int(r["count_a_to_b"]),
                count_b_to_a=int(r["count_b_to_a"]),
                line_role=r["line_role"] if r["line_role"] in ("both", "count", "speed") else "both",
            )
            for r in rows
        ]

    def replace_counting_lines(self, lines: list[CountingLineModel]) -> list[CountingLineModel]:
        with self._lock:
            existing = {
                r["id"]: (int(r["count_a_to_b"]), int(r["count_b_to_a"]))
                for r in self._conn.execute("SELECT id, count_a_to_b, count_b_to_a FROM counting_lines").fetchall()
            }
            cur = self._conn.cursor()
            cur.execute("DELETE FROM counting_lines")
            result: list[CountingLineModel] = []
            for line in lines:
                if line.line_role == "speed":
                    a_to_b, b_to_a = 0, 0
                else:
                    a_to_b, b_to_a = existing.get(line.id, (line.count_a_to_b, line.count_b_to_a))
                cur.execute(
                    """
                    INSERT INTO counting_lines (id, name, p1_json, p2_json, enabled, color, count_a_to_b, count_b_to_a, line_role)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        line.id,
                        line.name,
                        json.dumps(line.p1),
                        json.dumps(line.p2),
                        int(line.enabled),
                        line.color,
                        a_to_b,
                        b_to_a,
                        line.line_role,
                    ),
                )
                result.append(line.model_copy(update={"count_a_to_b": a_to_b, "count_b_to_a": b_to_a}))
            self._conn.commit()
        return result

    def increment_line_counts(self, increments: dict[str, tuple[int, int]]) -> None:
        if not increments:
            return
        with self._lock:
            for line_id, (a_to_b, b_to_a) in increments.items():
                if a_to_b or b_to_a:
                    self._conn.execute(
                        """
                        UPDATE counting_lines
                        SET count_a_to_b = count_a_to_b + ?,
                            count_b_to_a = count_b_to_a + ?
                        WHERE id = ?
                        """,
                        (a_to_b, b_to_a, line_id),
                    )
            self._conn.commit()

    def reset_line_counts(self) -> None:
        with self._lock:
            self._conn.execute("UPDATE counting_lines SET count_a_to_b = 0, count_b_to_a = 0")
            self._conn.commit()

    # --- Violations ---

    def add_violation(
        self,
        track_id: int,
        zone_id: str,
        zone_name: str,
        vehicle_class: str,
        started_at: str,
        flagged_at: str,
        dwell_seconds: float,
        snapshot_filename: str,
        license_plate: str = "",
    ) -> ViolationRecord:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO violations (
                    track_id, zone_id, zone_name, vehicle_class,
                    started_at, flagged_at, dwell_seconds, snapshot_filename, license_plate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    track_id,
                    zone_id,
                    zone_name,
                    vehicle_class,
                    started_at,
                    flagged_at,
                    dwell_seconds,
                    snapshot_filename,
                    license_plate,
                ),
            )
            self._conn.commit()
            row_id = int(cur.lastrowid)
        return ViolationRecord(
            id=row_id,
            track_id=track_id,
            zone_id=zone_id,
            zone_name=zone_name,
            vehicle_class=vehicle_class,
            started_at=started_at,
            flagged_at=flagged_at,
            dwell_seconds=dwell_seconds,
            snapshot_filename=snapshot_filename,
            license_plate=license_plate,
        )

    def update_violation_dwell(
        self,
        violation_id: int,
        dwell_seconds: float,
        *,
        flagged_at: str | None = None,
        license_plate: str | None = None,
    ) -> None:
        """Update the recorded parked duration (and optional latest timestamp/plate) for an existing violation."""
        sets = ["dwell_seconds = ?"]
        params: list[Any] = [round(float(dwell_seconds), 1)]
        if flagged_at is not None and flagged_at.strip():
            sets.append("flagged_at = ?")
            params.append(flagged_at.strip())
        if license_plate is not None and license_plate.strip():
            sets.append("license_plate = ?")
            params.append(license_plate.strip())
        params.append(int(violation_id))
        with self._lock:
            self._conn.execute(
                f"UPDATE violations SET {', '.join(sets)} WHERE id = ?",
                tuple(params),
            )
            self._conn.commit()

    def get_violations(self, limit: int = 50, offset: int = 0) -> list[ViolationRecord]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT id, track_id, zone_id, zone_name, vehicle_class,
                       started_at, flagged_at, dwell_seconds, snapshot_filename, license_plate
                FROM violations
                ORDER BY id DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [ViolationRecord(**dict(r)) for r in rows]

    def delete_violation(self, violation_id: int) -> str | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT snapshot_filename FROM violations WHERE id = ?", (violation_id,)
            ).fetchone()
            if not row:
                return None
            self._conn.execute("DELETE FROM violations WHERE id = ?", (violation_id,))
            self._conn.commit()
            return str(row["snapshot_filename"])

    def clear_violations(self) -> list[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT snapshot_filename FROM violations"
            ).fetchall()
            self._conn.execute("DELETE FROM violations")
            self._conn.commit()
            return [str(r["snapshot_filename"]) for r in rows if r["snapshot_filename"]]

    def count_violations(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) AS c FROM violations").fetchone()
            return int(row["c"]) if row else 0

    # --- Stats & Hourly Traffic ---

    def get_stats_state(self) -> dict[str, int]:
        with self._lock:
            row = self._conn.execute(
                "SELECT unique_cars, total_cars, unique_people, total_people, person_alerts FROM stats_state WHERE id = 1"
            ).fetchone()
        if not row:
            return {
                "unique_cars": 0,
                "total_cars": 0,
                "unique_people": 0,
                "total_people": 0,
                "person_alerts": 0,
            }
        return dict(row)

    def increment_stats(
        self,
        unique_cars: int = 0,
        total_cars: int = 0,
        unique_people: int = 0,
        total_people: int = 0,
        person_alerts: int = 0,
        now: datetime | None = None,
    ) -> dict[str, int]:
        ts = now or datetime.now(timezone.utc)
        hour_bucket = ts.astimezone().strftime("%Y-%m-%dT%H")
        with self._lock:
            self._conn.execute(
                """
                UPDATE stats_state
                SET unique_cars = unique_cars + ?,
                    total_cars = total_cars + ?,
                    unique_people = unique_people + ?,
                    total_people = total_people + ?,
                    person_alerts = person_alerts + ?
                WHERE id = 1
                """,
                (unique_cars, total_cars, unique_people, total_people, person_alerts),
            )
            if total_cars or total_people:
                self._conn.execute(
                    """
                    INSERT INTO hourly_traffic (hour_bucket, vehicles_count, people_count)
                    VALUES (?, ?, ?)
                    ON CONFLICT(hour_bucket) DO UPDATE SET
                        vehicles_count = vehicles_count + excluded.vehicles_count,
                        people_count = people_count + excluded.people_count
                    """,
                    (hour_bucket, total_cars, total_people),
                )
            self._conn.commit()
            row = self._conn.execute(
                "SELECT unique_cars, total_cars, unique_people, total_people, person_alerts FROM stats_state WHERE id = 1"
            ).fetchone()
        return dict(row)

    def reset_stats(self) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE stats_state SET unique_cars = 0, total_cars = 0, unique_people = 0, total_people = 0, person_alerts = 0 WHERE id = 1"
            )
            self._conn.execute("DELETE FROM hourly_traffic")
            self._conn.commit()

    def get_hourly_traffic(self, limit: int = 24) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT hour_bucket, vehicles_count, people_count
                FROM hourly_traffic
                ORDER BY hour_bucket DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    # --- Runtime Config ---

    def get_saved_config(self) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT config_json FROM app_config WHERE id = 1").fetchone()
        if not row:
            return None
        return json.loads(row["config_json"])

    def save_config(self, config_dict: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO app_config (id, config_json) VALUES (1, ?)
                ON CONFLICT(id) DO UPDATE SET config_json = excluded.config_json
                """,
                (json.dumps(config_dict),),
            )
            self._conn.commit()

    def get_meta(self, key: str) -> str | None:
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta_state WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO meta_state (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value
                """,
                (key, value),
            )
            self._conn.commit()

    def get_ground_calibration(self) -> GroundCalibrationModel | None:
        raw = self.get_meta("ground_calibration")
        if not raw:
            return None
        try:
            return GroundCalibrationModel(**json.loads(raw))
        except Exception:
            return None

    def save_ground_calibration(self, calib: GroundCalibrationModel) -> GroundCalibrationModel:
        self.set_meta("ground_calibration", calib.model_dump_json())
        return calib

    def delete_ground_calibration(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM meta_state WHERE key = 'ground_calibration'")
            self._conn.commit()

    def get_daily_summary(self, date_str: str | None = None) -> dict[str, Any]:
        """Aggregate traffic, peak hour, line crossings, and violations for a local YYYY-MM-DD date."""
        target_date = date_str or datetime.now().strftime("%Y-%m-%d")
        with self._lock:
            hour_rows = self._conn.execute(
                """
                SELECT hour_bucket, vehicles_count, people_count
                FROM hourly_traffic
                WHERE hour_bucket LIKE ?
                ORDER BY hour_bucket ASC
                """,
                (f"{target_date}T%",),
            ).fetchall()
            violation_rows = self._conn.execute(
                """
                SELECT zone_name, vehicle_class, flagged_at, dwell_seconds, license_plate
                FROM violations
                ORDER BY id DESC
                LIMIT 500
                """
            ).fetchall()
        hours = [dict(r) for r in hour_rows]
        vehicles_today = sum(int(r["vehicles_count"]) for r in hours)
        people_today = sum(int(r["people_count"]) for r in hours)
        peak_hour = max(hours, key=lambda r: int(r["vehicles_count"]), default=None)

        violations_today: list[dict[str, Any]] = []
        for vr in violation_rows:
            try:
                dt = datetime.fromisoformat(str(vr["flagged_at"]))
                local_date = dt.astimezone().strftime("%Y-%m-%d") if dt.tzinfo else dt.strftime("%Y-%m-%d")
            except ValueError:
                local_date = str(vr["flagged_at"])[:10]
            if local_date == target_date:
                violations_today.append(dict(vr))

        stats = self.get_stats_state()
        lines = [ln.model_dump() for ln in self.get_counting_lines()]
        return {
            "date": target_date,
            "vehicles_today": vehicles_today,
            "people_today": people_today,
            "peak_hour": peak_hour,
            "violations_today": len(violations_today),
            "person_alerts_today": sum(
                1 for v in violations_today if v.get("vehicle_class") == "person"
            ),
            "violations_list": violations_today[:10],
            "counting_lines": lines,
            "cumulative_stats": stats,
        }
