from __future__ import annotations

import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import types
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.database import CountingLineModel, GroundCalibrationModel, ZoneModel

VEHICLE_CLASSES: dict[int, str] = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}
PERSON_CLASS_ID = 0
TARGET_CLASS_IDS = [PERSON_CLASS_ID, *VEHICLE_CLASSES.keys()]


@dataclass
class TrackedDetection:
    """Single tracked object in pixel coordinates."""

    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    is_emergency: bool = False

    @property
    def is_vehicle(self) -> bool:
        return self.class_id in VEHICLE_CLASSES

    @property
    def is_person(self) -> bool:
        return self.class_id == PERSON_CLASS_ID

    def bottom_center_norm(self, width: int, height: int) -> tuple[float, float]:
        x1, _, x2, y2 = self.bbox_xyxy
        cx = ((x1 + x2) * 0.5) / max(width, 1)
        cy = y2 / max(height, 1)
        return (min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0))


@dataclass
class DwellState:
    """Tracks how long a vehicle has been stationary inside a specific zone."""

    zone_id: str
    zone_name: str
    entered_at: float
    last_seen_at: float
    anchor_pos: tuple[float, float]
    stationary_seconds: float = 0.0
    flagged: bool = False
    violation_key: tuple[int, str] | None = None


@dataclass
class ViolationEvent:
    """Emitted when a vehicle exceeds the stationary dwell threshold in a red zone."""

    track_id: int
    zone_id: str
    zone_name: str
    vehicle_class: str
    started_at: str
    flagged_at: str
    dwell_seconds: float


@dataclass
class ViolationDwellUpdate:
    """Live or final dwell duration update for an already-flagged red-zone violation."""

    violation_key: tuple[int, str]
    track_id: int
    zone_id: str
    dwell_seconds: float
    completed: bool = False


@dataclass
class MonitorStepResult:
    """Output of a single frame update in RedZoneMonitor."""

    active_cars: int = 0
    active_people: int = 0
    new_unique_cars: int = 0
    new_total_cars: int = 0
    new_unique_people: int = 0
    new_total_people: int = 0
    new_person_alerts: int = 0
    new_violations: list[ViolationEvent] = field(default_factory=list)
    dwell_updates: list[ViolationDwellUpdate] = field(default_factory=list)
    track_statuses: dict[int, dict[str, Any]] = field(default_factory=dict)


def point_in_normalized_polygon(nx: float, ny: float, points: list[list[float]]) -> bool:
    """Return True if normalized point (nx, ny) is inside or on the normalized polygon."""
    if len(points) < 3:
        return False
    contour = np.array([[p[0] * 1000.0, p[1] * 1000.0] for p in points], dtype=np.float32)
    result = cv2.pointPolygonTest(contour, (nx * 1000.0, ny * 1000.0), False)
    return result >= 0


class HotspotTracker:
    """Accumulates spatial traffic density on a fixed normalized grid."""

    def __init__(self, grid_w: int = 320, grid_h: int = 180, decay: float = 0.9985) -> None:
        self.grid_w = grid_w
        self.grid_h = grid_h
        self.decay = decay
        self.grid = np.zeros((grid_h, grid_w), dtype=np.float32)

    def update(self, detections: list[TrackedDetection], frame_w: int, frame_h: int) -> None:
        self.grid *= self.decay
        for det in detections:
            nx, ny = det.bottom_center_norm(frame_w, frame_h)
            gx = min(max(int(nx * (self.grid_w - 1)), 0), self.grid_w - 1)
            gy = min(max(int(ny * (self.grid_h - 1)), 0), self.grid_h - 1)
            self.grid[gy, gx] += 1.0

    def reset(self) -> None:
        self.grid.fill(0.0)

    def render_overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        max_val = float(self.grid.max())
        if max_val <= 0.05:
            return frame

        blurred = cv2.GaussianBlur(self.grid, (15, 15), 0)
        norm = np.clip((blurred / max(float(blurred.max()), 1e-5)) * 255.0, 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(norm, cv2.COLORMAP_INFERNO)
        h, w = frame.shape[:2]
        colored_resized = cv2.resize(colored, (w, h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(norm, (w, h), interpolation=cv2.INTER_LINEAR) > 12

        out = frame.copy()
        blended = cv2.addWeighted(frame, 1.0 - alpha, colored_resized, alpha, 0)
        out[mask] = blended[mask]
        return out


class EmergencyVehicleClassifier:
    """Identifies emergency vehicles (police, fire, ambulance) via lightbars & fire-engine livery."""

    def __init__(self) -> None:
        self._confirmed_ids: set[int] = set()
        self._roof_signal_history: dict[int, list[float]] = {}

    def reset(self) -> None:
        self._confirmed_ids.clear()
        self._roof_signal_history.clear()

    def update_detections(
        self,
        frame: np.ndarray,
        tracks: list[TrackedDetection],
        min_saturation: int = 190,
        min_value: int = 225,
        strobe_swing: float = 0.015,
        fire_red_ratio: float = 0.35,
    ) -> None:
        h, w = frame.shape[:2]
        active_ids = {t.track_id for t in tracks}
        for stale_id in list(self._roof_signal_history.keys()):
            if stale_id not in active_ids:
                self._roof_signal_history.pop(stale_id, None)
                self._confirmed_ids.discard(stale_id)

        for det in tracks:
            if not det.is_vehicle:
                continue
            if det.track_id in self._confirmed_ids:
                det.is_emergency = True
                continue
            if self._classify_crop(
                frame, det, w, h, min_saturation, min_value, strobe_swing, fire_red_ratio
            ):
                self._confirmed_ids.add(det.track_id)
                det.is_emergency = True

    def _classify_crop(
        self,
        frame: np.ndarray,
        det: TrackedDetection,
        w: int,
        h: int,
        min_saturation: int = 190,
        min_value: int = 225,
        strobe_swing: float = 0.015,
        fire_red_ratio: float = 0.35,
    ) -> bool:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if (x2 - x1) < 16 or (y2 - y1) < 16:
            return False

        crop = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # 1. Check upper 28% (roof / lightbar area) for bright glowing red+blue lightbars or flashing strobes
        roof_h = max(4, int(crop.shape[0] * 0.28))
        roof_hsv = hsv[:roof_h, :]
        h_ch, s_ch, v_ch = roof_hsv[:, :, 0], roof_hsv[:, :, 1], roof_hsv[:, :, 2]
        roof_pixels = float(roof_hsv.shape[0] * roof_hsv.shape[1])

        bright_red = ((h_ch <= 10) | (h_ch >= 170)) & (s_ch >= min_saturation) & (v_ch >= min_value)
        bright_blue = (h_ch >= 100) & (h_ch <= 130) & (s_ch >= min_saturation) & (v_ch >= min_value)
        red_roof_ratio = float(np.count_nonzero(bright_red)) / roof_pixels
        blue_roof_ratio = float(np.count_nonzero(bright_blue)) / roof_pixels
        strobe_ratio = red_roof_ratio + blue_roof_ratio

        # Dual red+blue lightbar (compact, not a solid-colored car)
        if red_roof_ratio >= 0.004 and blue_roof_ratio >= 0.004 and strobe_ratio <= 0.30:
            return True

        # Track temporal strobe variance across frames for flashing lights
        hist = self._roof_signal_history.setdefault(det.track_id, [])
        hist.append(strobe_ratio)
        if len(hist) > 10:
            hist.pop(0)
        if len(hist) >= 3 and (max(hist) - min(hist)) >= strobe_swing and max(hist) >= strobe_swing:
            return True

        # 2. Fire truck dominant vivid red body on trucks/buses
        if det.class_name in ("truck", "bus"):
            body_h, body_s, body_v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
            fire_red = ((body_h <= 10) | (body_h >= 170)) & (body_s >= 140) & (body_v >= 110)
            red_ratio = float(np.count_nonzero(fire_red)) / float(hsv.shape[0] * hsv.shape[1])
            if red_ratio >= fire_red_ratio:
                return True

        return False


def filter_excluded_tracks(
    tracks: list[TrackedDetection],
    frame_w: int,
    frame_h: int,
    zones: list[ZoneModel],
) -> list[TrackedDetection]:
    """Remove detections whose bottom-center falls inside any enabled exclusion mask."""
    exclusions = [z for z in zones if z.enabled and z.zone_type == "exclusion" and len(z.points) >= 3]
    if not exclusions:
        return tracks
    kept: list[TrackedDetection] = []
    for det in tracks:
        nx, ny = det.bottom_center_norm(frame_w, frame_h)
        if any(point_in_normalized_polygon(nx, ny, ex.points) for ex in exclusions):
            continue
        kept.append(det)
    return kept


def parse_hhmm_minutes(hhmm: str, default_minutes: int = 0) -> int:
    """Parse an HH:MM string into minutes since midnight [0..1439]."""
    raw = str(hhmm or "").strip()
    if ":" not in raw:
        return default_minutes
    parts = raw.split(":")
    try:
        hour = int(parts[0])
        minute = int(parts[1])
    except (ValueError, IndexError):
        return default_minutes
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return default_minutes
    return hour * 60 + minute


def is_within_time_window(
    now_ts: float,
    start_hhmm: str,
    end_hhmm: str,
    local_tz: Any = None,
) -> bool:
    """Return True when wall-clock time falls inside [start_hhmm, end_hhmm).

    Supports both same-day windows (e.g. 09:00-17:00) and overnight
    cross-midnight windows (e.g. 00:00-05:00 or 23:00-05:00). Equal start/end
    means an active 24-hour window.
    """
    start_min = parse_hhmm_minutes(start_hhmm, 0)
    end_min = parse_hhmm_minutes(end_hhmm, 5 * 60)
    dt = datetime.fromtimestamp(now_ts, tz=local_tz)
    cur_min = dt.hour * 60 + dt.minute
    if start_min == end_min:
        return True
    if start_min < end_min:
        return start_min <= cur_min < end_min
    return cur_min >= start_min or cur_min < end_min


class RedZoneMonitor:
    """Counts unique/total vehicles & people and flags illegal red-zone parking."""

    def __init__(self) -> None:
        self._seen_frames: dict[int, int] = {}
        self._counted_tracks: set[int] = set()
        self._last_seen_ts: dict[int, float] = {}
        self._initial_pos: dict[int, tuple[float, float]] = {}
        self._counted_vehicle_pos: dict[int, tuple[float, float]] = {}
        self._dwell_states: dict[tuple[int, str], DwellState] = {}
        self._flagged_pairs: set[tuple[int, str]] = set()
        self._person_dwell_states: dict[tuple[int, str], DwellState] = {}
        self._flagged_person_pairs: set[tuple[int, str]] = set()

    def reset(self) -> None:
        self._seen_frames.clear()
        self._counted_tracks.clear()
        self._last_seen_ts.clear()
        self._initial_pos.clear()
        self._counted_vehicle_pos.clear()
        self._dwell_states.clear()
        self._flagged_pairs.clear()
        self._person_dwell_states.clear()
        self._flagged_person_pairs.clear()

    def process_tracks(
        self,
        tracks: list[TrackedDetection],
        frame_w: int,
        frame_h: int,
        zones: list[ZoneModel],
        now_ts: float,
        dwell_seconds: float,
        movement_threshold: float,
        min_track_frames: int,
        exempt_emergency_vehicles: bool = True,
        dwell_grace_seconds: float = 5.0,
        person_zone_enabled: bool = False,
        person_zone_start_time: str = "00:00",
        person_zone_end_time: str = "05:00",
        person_zone_dwell_seconds: float = 2.0,
        person_zone_scope: str = "person_zones_or_frame",
        local_tz: Any = None,
    ) -> MonitorStepResult:
        res = MonitorStepResult()
        enabled_zones = [
            z for z in zones if z.enabled and z.zone_type == "red_zone" and len(z.points) >= 3
        ]
        active_pairs: set[tuple[int, str]] = set()
        active_vehicle_ids = {d.track_id for d in tracks if d.is_vehicle}

        person_window_active = bool(
            person_zone_enabled
            and is_within_time_window(
                now_ts,
                person_zone_start_time,
                person_zone_end_time,
                local_tz=local_tz,
            )
        )
        enabled_person_zones = [
            z for z in zones if z.enabled and z.zone_type == "person_zone" and len(z.points) >= 3
        ]
        person_targets: list[tuple[str, str, str, list[list[float]] | None]] = []
        if person_window_active:
            if person_zone_scope == "person_zones":
                for z in enabled_person_zones:
                    person_targets.append((f"person_zone:{z.id}", f"person_zone:{z.id}", z.name, z.points))
            elif person_zone_scope == "red_zones":
                for z in enabled_zones:
                    person_targets.append((f"person_zone:{z.id}", f"person_zone:{z.id}", z.name, z.points))
            elif person_zone_scope == "full_frame":
                person_targets.append(
                    ("person_zone:after_hours", "person_zone:after_hours", "After-Hours Watch", None)
                )
            else:  # "person_zones_or_frame"
                if enabled_person_zones:
                    for z in enabled_person_zones:
                        person_targets.append(
                            (f"person_zone:{z.id}", f"person_zone:{z.id}", z.name, z.points)
                        )
                else:
                    person_targets.append(
                        ("person_zone:after_hours", "person_zone:after_hours", "After-Hours Watch", None)
                    )
        active_person_pairs: set[tuple[int, str]] = set()
        active_person_ids = {d.track_id for d in tracks if d.is_person}

        for det in tracks:
            tid = det.track_id
            self._last_seen_ts[tid] = now_ts
            self._seen_frames[tid] = self._seen_frames.get(tid, 0) + 1
            nx, ny = (
                det.bottom_center_norm(frame_w, frame_h)
                if (det.is_vehicle or det.is_person)
                else (0.0, 0.0)
            )
            if det.is_vehicle:
                self._initial_pos.setdefault(tid, (nx, ny))

            if det.is_vehicle:
                res.active_cars += 1
            elif det.is_person:
                res.active_people += 1

            if tid not in self._counted_tracks and self._seen_frames[tid] >= min_track_frames:
                self._counted_tracks.add(tid)
                if det.is_vehicle:
                    origin = self._initial_pos.get(tid, (nx, ny))
                    barely_moved = math.hypot(nx - origin[0], ny - origin[1]) <= max(
                        movement_threshold * 1.5, 0.025
                    )
                    dedup_radius = max(movement_threshold * 2.5, 0.035)
                    is_static_dup = False
                    if barely_moved:
                        for other_tid, other_pos in self._counted_vehicle_pos.items():
                            if other_tid == tid or other_tid in active_vehicle_ids:
                                continue
                            if (now_ts - self._last_seen_ts.get(other_tid, 0.0)) > 300.0:
                                continue
                            if math.hypot(nx - other_pos[0], ny - other_pos[1]) <= dedup_radius:
                                is_static_dup = True
                                break
                    if not is_static_dup:
                        res.new_unique_cars += 1
                        res.new_total_cars += 1
                elif det.is_person:
                    res.new_unique_people += 1
                    res.new_total_people += 1

            if det.is_vehicle and tid in self._counted_tracks:
                self._counted_vehicle_pos[tid] = (nx, ny)

            status: dict[str, Any] = {
                "in_red_zone": False,
                "dwell_seconds": 0.0,
                "violation": False,
                "zone_name": None,
                "emergency": det.is_emergency,
                "in_person_zone": False,
                "person_dwell_seconds": 0.0,
                "person_violation": False,
            }

            if det.is_vehicle and enabled_zones and not (exempt_emergency_vehicles and det.is_emergency):
                for zone in enabled_zones:
                    if not point_in_normalized_polygon(nx, ny, zone.points):
                        continue

                    pair = (tid, zone.id)
                    active_pairs.add(pair)
                    state = self._dwell_states.get(pair)
                    if state is None:
                        # Recover an active flagged dwell session if tracker reassigned the vehicle's track ID at the same spot
                        transfer_radius = max(movement_threshold * 3.0, 0.05)
                        transfer_grace = max(dwell_grace_seconds, 15.0)
                        best_prev_pair: tuple[int, str] | None = None
                        best_prev_dist = transfer_radius
                        for prev_pair, prev_state in self._dwell_states.items():
                            if (
                                prev_pair[1] != zone.id
                                or prev_pair in active_pairs
                                or prev_pair[0] in active_vehicle_ids
                                or not prev_state.flagged
                                or prev_state.violation_key is None
                                or (now_ts - prev_state.last_seen_at) > transfer_grace
                            ):
                                continue
                            dist_prev = math.hypot(nx - prev_state.anchor_pos[0], ny - prev_state.anchor_pos[1])
                            if dist_prev <= best_prev_dist:
                                best_prev_dist = dist_prev
                                best_prev_pair = prev_pair

                        if best_prev_pair is not None:
                            state = self._dwell_states.pop(best_prev_pair)
                            self._flagged_pairs.discard(best_prev_pair)
                            self._flagged_pairs.add(pair)
                            dt = max(0.0, now_ts - state.last_seen_at)
                            state.stationary_seconds = max(
                                state.stationary_seconds + dt, now_ts - state.entered_at
                            )
                            state.anchor_pos = (nx, ny)
                            state.last_seen_at = now_ts
                            self._dwell_states[pair] = state
                        else:
                            state = DwellState(
                                zone_id=zone.id,
                                zone_name=zone.name,
                                entered_at=now_ts,
                                last_seen_at=now_ts,
                                anchor_pos=(nx, ny),
                                stationary_seconds=0.0,
                                flagged=(pair in self._flagged_pairs),
                            )
                            self._dwell_states[pair] = state
                    else:
                        dt = max(0.0, now_ts - state.last_seen_at)
                        dist = math.hypot(nx - state.anchor_pos[0], ny - state.anchor_pos[1])
                        if state.flagged:
                            # Keep counting continuously once a red-zone violation is active
                            state.stationary_seconds = max(
                                state.stationary_seconds + dt, now_ts - state.entered_at
                            )
                            state.anchor_pos = (nx, ny)
                        elif dist <= movement_threshold:
                            state.stationary_seconds += dt
                        else:
                            # Vehicle is actively moving before threshold; reset stationary timer
                            state.anchor_pos = (nx, ny)
                            state.stationary_seconds = 0.0
                            state.entered_at = now_ts
                        state.last_seen_at = now_ts

                    newly_flagged = False
                    if state.stationary_seconds >= dwell_seconds and not state.flagged:
                        state.flagged = True
                        state.violation_key = (tid, zone.id)
                        self._flagged_pairs.add(pair)
                        newly_flagged = True
                        started_iso = datetime.fromtimestamp(state.entered_at, tz=timezone.utc).astimezone().isoformat()
                        flagged_iso = datetime.fromtimestamp(now_ts, tz=timezone.utc).astimezone().isoformat()
                        res.new_violations.append(
                            ViolationEvent(
                                track_id=tid,
                                zone_id=zone.id,
                                zone_name=zone.name,
                                vehicle_class=det.class_name,
                                started_at=started_iso,
                                flagged_at=flagged_iso,
                                dwell_seconds=round(state.stationary_seconds, 1),
                            )
                        )

                    if state.flagged and not newly_flagged and state.violation_key is not None:
                        res.dwell_updates.append(
                            ViolationDwellUpdate(
                                violation_key=state.violation_key,
                                track_id=tid,
                                zone_id=zone.id,
                                dwell_seconds=round(state.stationary_seconds, 1),
                                completed=False,
                            )
                        )

                    status["in_red_zone"] = True
                    status["zone_name"] = zone.name
                    status["dwell_seconds"] = max(status["dwell_seconds"], round(state.stationary_seconds, 1))
                    status["violation"] = status["violation"] or state.flagged

            if det.is_person and person_targets:
                for target_key, zid, zname, zpoints in person_targets:
                    if zpoints is not None and not point_in_normalized_polygon(nx, ny, zpoints):
                        continue

                    ppair = (tid, target_key)
                    active_person_pairs.add(ppair)
                    pstate = self._person_dwell_states.get(ppair)
                    if pstate is None:
                        transfer_radius = max(movement_threshold * 5.0, 0.08)
                        transfer_grace = max(dwell_grace_seconds, 15.0)
                        best_prev_ppair: tuple[int, str] | None = None
                        best_prev_pdist = transfer_radius
                        for prev_pair, prev_state in self._person_dwell_states.items():
                            if (
                                prev_pair[1] != target_key
                                or prev_pair in active_person_pairs
                                or prev_pair[0] in active_person_ids
                                or not prev_state.flagged
                                or prev_state.violation_key is None
                                or (now_ts - prev_state.last_seen_at) > transfer_grace
                            ):
                                continue
                            dist_prev = math.hypot(
                                nx - prev_state.anchor_pos[0], ny - prev_state.anchor_pos[1]
                            )
                            if dist_prev <= best_prev_pdist:
                                best_prev_pdist = dist_prev
                                best_prev_ppair = prev_pair

                        if best_prev_ppair is not None:
                            pstate = self._person_dwell_states.pop(best_prev_ppair)
                            self._flagged_person_pairs.discard(best_prev_ppair)
                            self._flagged_person_pairs.add(ppair)
                            dt = max(0.0, now_ts - pstate.last_seen_at)
                            pstate.stationary_seconds = max(
                                pstate.stationary_seconds + dt, now_ts - pstate.entered_at
                            )
                            pstate.anchor_pos = (nx, ny)
                            pstate.last_seen_at = now_ts
                            self._person_dwell_states[ppair] = pstate
                        else:
                            pstate = DwellState(
                                zone_id=zid,
                                zone_name=zname,
                                entered_at=now_ts,
                                last_seen_at=now_ts,
                                anchor_pos=(nx, ny),
                                stationary_seconds=0.0,
                                flagged=(ppair in self._flagged_person_pairs),
                            )
                            self._person_dwell_states[ppair] = pstate
                    else:
                        dt = max(0.0, now_ts - pstate.last_seen_at)
                        # Unlike vehicles, people walk continuously inside watched areas
                        pstate.stationary_seconds += dt
                        pstate.anchor_pos = (nx, ny)
                        pstate.last_seen_at = now_ts

                    newly_flagged_person = False
                    if pstate.stationary_seconds >= person_zone_dwell_seconds and not pstate.flagged:
                        pstate.flagged = True
                        pstate.violation_key = (tid, zid)
                        self._flagged_person_pairs.add(ppair)
                        newly_flagged_person = True
                        res.new_person_alerts += 1
                        started_iso = (
                            datetime.fromtimestamp(pstate.entered_at, tz=timezone.utc)
                            .astimezone()
                            .isoformat()
                        )
                        flagged_iso = (
                            datetime.fromtimestamp(now_ts, tz=timezone.utc)
                            .astimezone()
                            .isoformat()
                        )
                        res.new_violations.append(
                            ViolationEvent(
                                track_id=tid,
                                zone_id=zid,
                                zone_name=zname,
                                vehicle_class="person",
                                started_at=started_iso,
                                flagged_at=flagged_iso,
                                dwell_seconds=round(pstate.stationary_seconds, 1),
                            )
                        )

                    if pstate.flagged and not newly_flagged_person and pstate.violation_key is not None:
                        res.dwell_updates.append(
                            ViolationDwellUpdate(
                                violation_key=pstate.violation_key,
                                track_id=tid,
                                zone_id=zid,
                                dwell_seconds=round(pstate.stationary_seconds, 1),
                                completed=False,
                            )
                        )

                    status["in_person_zone"] = True
                    if not status["zone_name"]:
                        status["zone_name"] = zname
                    status["person_dwell_seconds"] = max(
                        status["person_dwell_seconds"], round(pstate.stationary_seconds, 1)
                    )
                    status["person_violation"] = status["person_violation"] or pstate.flagged

            res.track_statuses[tid] = status

        # Clear dwell states only after exceeding the dropout grace period; finalize flagged duration on departure
        for pair in list(self._dwell_states.keys()):
            if pair in active_pairs:
                continue
            st = self._dwell_states[pair]
            if st.flagged:
                grace = (
                    min(dwell_grace_seconds, 2.0)
                    if pair[0] in active_vehicle_ids
                    else max(dwell_grace_seconds, 15.0)
                )
            else:
                grace = dwell_grace_seconds
            if (now_ts - st.last_seen_at) > grace:
                if st.flagged and st.violation_key is not None:
                    res.dwell_updates.append(
                        ViolationDwellUpdate(
                            violation_key=st.violation_key,
                            track_id=pair[0],
                            zone_id=pair[1],
                            dwell_seconds=round(st.stationary_seconds, 1),
                            completed=True,
                        )
                    )
                self._flagged_pairs.discard(pair)
                del self._dwell_states[pair]

        for ppair in list(self._person_dwell_states.keys()):
            if ppair in active_person_pairs:
                continue
            pst = self._person_dwell_states[ppair]
            if pst.flagged:
                p_grace = (
                    min(dwell_grace_seconds, 2.0)
                    if ppair[0] in active_person_ids
                    else max(dwell_grace_seconds, 15.0)
                )
            else:
                p_grace = dwell_grace_seconds
            if not person_window_active or (now_ts - pst.last_seen_at) > p_grace:
                if pst.flagged and pst.violation_key is not None:
                    res.dwell_updates.append(
                        ViolationDwellUpdate(
                            violation_key=pst.violation_key,
                            track_id=ppair[0],
                            zone_id=ppair[1],
                            dwell_seconds=round(pst.stationary_seconds, 1),
                            completed=True,
                        )
                    )
                self._flagged_person_pairs.discard(ppair)
                del self._person_dwell_states[ppair]

        # Retain track metadata for 15 minutes so brief dropouts never inflate unique vehicle counts
        stale_tids = [tid for tid, ts in self._last_seen_ts.items() if (now_ts - ts) > 900.0]
        for tid in stale_tids:
            self._last_seen_ts.pop(tid, None)
            self._seen_frames.pop(tid, None)
            self._initial_pos.pop(tid, None)
            self._counted_vehicle_pos.pop(tid, None)
            self._counted_tracks.discard(tid)

        return res


def _ccw(ax: float, ay: float, bx: float, by: float, cx: float, cy: float) -> float:
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def _segments_intersect(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> bool:
    """Return True if 2D line segments a1-a2 and b1-b2 intersect."""
    d1 = _ccw(b1[0], b1[1], b2[0], b2[1], a1[0], a1[1])
    d2 = _ccw(b1[0], b1[1], b2[0], b2[1], a2[0], a2[1])
    d3 = _ccw(a1[0], a1[1], a2[0], a2[1], b1[0], b1[1])
    d4 = _ccw(a1[0], a1[1], a2[0], a2[1], b2[0], b2[1])
    return ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    )


def _segment_intersection_point(
    a1: tuple[float, float],
    a2: tuple[float, float],
    b1: tuple[float, float],
    b2: tuple[float, float],
) -> tuple[float, tuple[float, float]] | None:
    """Return (t_along_a, (ix, iy)) if segments a1-a2 and b1-b2 intersect, else None."""
    if not _segments_intersect(a1, a2, b1, b2):
        return None
    dx_a = a2[0] - a1[0]
    dy_a = a2[1] - a1[1]
    dx_b = b2[0] - b1[0]
    dy_b = b2[1] - b1[1]
    denom = dx_a * dy_b - dy_a * dx_b
    if abs(denom) < 1e-9:
        return 0.5, (0.5 * (a1[0] + a2[0]), 0.5 * (a1[1] + a2[1]))
    t = ((b1[0] - a1[0]) * dy_b - (b1[1] - a1[1]) * dx_b) / denom
    t = max(0.0, min(1.0, t))
    return t, (a1[0] + t * dx_a, a1[1] + t * dy_a)


def _perp_line_distance(
    pt: tuple[float, float], p1: tuple[float, float], p2: tuple[float, float]
) -> float:
    """Return perpendicular distance from normalized point `pt` to infinite line `p1`-`p2`."""
    seg_len = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if seg_len < 1e-6:
        return math.hypot(pt[0] - p1[0], pt[1] - p1[1])
    return abs(_ccw(p1[0], p1[1], p2[0], p2[1], pt[0], pt[1])) / seg_len


class LineCrossingMonitor:
    """Counts vehicles crossing user-defined traffic flow lines and estimates two-line speed."""

    def __init__(self) -> None:
        self._prev_pos: dict[int, tuple[float, float]] = {}
        self._origin_pos: dict[int, tuple[float, float]] = {}
        self._prev_ts: dict[int, float] = {}
        self._last_seen_ts: dict[int, float] = {}
        self._crossed_pairs: set[tuple[int, str]] = set()
        self._pending_crossings: dict[tuple[int, str], tuple[float, tuple[float, float], float]] = {}
        self._first_crossing: dict[int, tuple[str, str, float, tuple[float, float]]] = {}
        self._track_speeds_mph: dict[int, float] = {}
        self._speed_anchor: dict[int, tuple[float, float]] = {}
        self._stationary_since: dict[int, float] = {}
        self._recent_speeds: deque[dict[str, Any]] = deque(maxlen=25)
        self._new_speed_events: list[dict[str, Any]] = []
        self._calibrated: bool = False

    def reset(self) -> None:
        self._prev_pos.clear()
        self._origin_pos.clear()
        self._prev_ts.clear()
        self._last_seen_ts.clear()
        self._crossed_pairs.clear()
        self._pending_crossings.clear()
        self._first_crossing.clear()
        self._track_speeds_mph.clear()
        self._speed_anchor.clear()
        self._stationary_since.clear()
        self._recent_speeds.clear()
        self._new_speed_events.clear()

    def consume_new_speeds(self) -> list[dict[str, Any]]:
        events = list(self._new_speed_events)
        self._new_speed_events.clear()
        return events

    def get_track_speeds(self) -> dict[int, float]:
        return dict(self._track_speeds_mph)

    def get_speed_summary(self) -> dict[str, Any]:
        recent = list(self._recent_speeds)
        speeds = [item["speed_mph"] for item in recent]
        return {
            "calibrated": self._calibrated,
            "last_speed_mph": speeds[-1] if speeds else None,
            "avg_speed_mph": round(sum(speeds) / len(speeds), 1) if speeds else None,
            "max_speed_mph": max(speeds) if speeds else None,
            "samples_count": len(speeds),
            "recent": list(reversed(recent[-8:])),
        }

    def _record_crossing(
        self,
        tid: int,
        det: TrackedDetection,
        ln: CountingLineModel,
        cross_ts: float,
        cross_pt: tuple[float, float],
        side_prev: float,
        tf: dict[str, Any] | None,
        now_ts: float,
        curr: tuple[float, float],
        increments: dict[str, tuple[int, int]],
        speed_line_ids: set[str] | frozenset[str] | None = None,
    ) -> None:
        self._crossed_pairs.add((tid, ln.id))
        role = getattr(ln, "line_role", "both")
        counts_flow = role in ("both", "count")
        measures_speed = role in ("both", "speed")

        if counts_flow:
            a_inc, b_inc = increments.get(ln.id, (0, 0))
            if side_prev < 0:
                increments[ln.id] = (a_inc + 1, b_inc)
            else:
                increments[ln.id] = (a_inc, b_inc + 1)

        if not measures_speed:
            return

        first = self._first_crossing.get(tid)
        valid_first = first is not None and (
            speed_line_ids is None or first[0] in speed_line_ids
        )
        if not valid_first:
            self._first_crossing[tid] = (ln.id, ln.name, cross_ts, cross_pt)
        elif first[0] != ln.id and tf is not None and tid not in self._track_speeds_mph:
            _, first_name, t1, pt1 = first
            dt_cross = cross_ts - t1
            g_pts = project_points([[pt1[0], pt1[1]], [cross_pt[0], cross_pt[1]]], tf["img_norm_to_ground"])
            if len(g_pts) == 2 and dt_cross >= 0.05:
                dist_ft = math.hypot(g_pts[1][0] - g_pts[0][0], g_pts[1][1] - g_pts[0][1])
                if dist_ft >= 1.0:
                    mph = (dist_ft / dt_cross) * 0.681818
                    if 0.5 <= mph <= 120.0:
                        mph_rounded = round(mph, 1)
                        self._track_speeds_mph[tid] = mph_rounded
                        self._speed_anchor[tid] = curr
                        self._stationary_since[tid] = now_ts
                        speed_event = {
                            "track_id": tid,
                            "vehicle_class": det.class_name,
                            "speed_mph": mph_rounded,
                            "from_line": first_name,
                            "to_line": ln.name,
                            "distance_ft": round(dist_ft, 1),
                            "dt_s": round(dt_cross, 2),
                        }
                        self._recent_speeds.append(speed_event)
                        self._new_speed_events.append(speed_event)

    def process_tracks(
        self,
        tracks: list[TrackedDetection],
        frame_w: int,
        frame_h: int,
        lines: list[CountingLineModel],
        now_ts: float,
        calib: GroundCalibrationModel | None = None,
        movement_threshold: float = 0.015,
        stationary_clear_seconds: float = 3.0,
    ) -> dict[str, tuple[int, int]]:
        """Return {line_id: (new_a_to_b, new_b_to_a)} and estimate two-line speed when calibrated."""
        enabled_lines = [ln for ln in lines if ln.enabled]
        speed_line_ids = frozenset(
            ln.id for ln in enabled_lines if getattr(ln, "line_role", "both") in ("both", "speed")
        )
        increments: dict[str, tuple[int, int]] = {}
        self._calibrated = calib is not None
        tf = compute_ground_transforms(calib) if calib is not None else None
        min_travel = max(movement_threshold * 1.5, 0.022)

        for det in tracks:
            if not det.is_vehicle:
                continue
            tid = det.track_id
            curr = det.bottom_center_norm(frame_w, frame_h)
            origin = self._origin_pos.setdefault(tid, curr)
            net_disp = math.hypot(curr[0] - origin[0], curr[1] - origin[1])
            prev = self._prev_pos.get(tid)
            prev_ts = self._prev_ts.get(tid, now_ts)
            self._prev_pos[tid] = curr
            self._prev_ts[tid] = now_ts
            self._last_seen_ts[tid] = now_ts

            if prev is None or not enabled_lines:
                continue

            for ln in enabled_lines:
                pair = (tid, ln.id)
                if pair in self._crossed_pairs:
                    continue
                p1 = (ln.p1[0], ln.p1[1])
                p2 = (ln.p2[0], ln.p2[1])

                pending = self._pending_crossings.get(pair)
                if pending is not None:
                    pend_ts, pend_pt, pend_side_prev = pending
                    side_curr = _ccw(p1[0], p1[1], p2[0], p2[1], curr[0], curr[1])
                    if side_curr * pend_side_prev > 0:
                        self._pending_crossings.pop(pair, None)
                    elif net_disp >= min_travel and _perp_line_distance(curr, p1, p2) >= 0.008:
                        self._pending_crossings.pop(pair, None)
                        self._record_crossing(
                            tid,
                            det,
                            ln,
                            pend_ts,
                            pend_pt,
                            pend_side_prev,
                            tf,
                            now_ts,
                            curr,
                            increments,
                            speed_line_ids=speed_line_ids,
                        )
                    continue

                hit = _segment_intersection_point(prev, curr, p1, p2)
                if hit is None:
                    continue
                t_frac, cross_pt = hit
                cross_ts = prev_ts + t_frac * max(0.0, now_ts - prev_ts)
                side_prev = _ccw(p1[0], p1[1], p2[0], p2[1], prev[0], prev[1])

                if net_disp >= min_travel:
                    self._record_crossing(
                        tid,
                        det,
                        ln,
                        cross_ts,
                        cross_pt,
                        side_prev,
                        tf,
                        now_ts,
                        curr,
                        increments,
                        speed_line_ids=speed_line_ids,
                    )
                else:
                    self._pending_crossings[pair] = (cross_ts, cross_pt, side_prev)

            if tid in self._track_speeds_mph:
                anchor = self._speed_anchor.get(tid, curr)
                if math.hypot(curr[0] - anchor[0], curr[1] - anchor[1]) > movement_threshold:
                    self._speed_anchor[tid] = curr
                    self._stationary_since[tid] = now_ts
                elif (now_ts - self._stationary_since.get(tid, now_ts)) >= stationary_clear_seconds:
                    self._track_speeds_mph.pop(tid, None)
                    self._speed_anchor.pop(tid, None)
                    self._stationary_since.pop(tid, None)

        stale_tids = [tid for tid, ts in self._last_seen_ts.items() if (now_ts - ts) > 900.0]
        for tid in stale_tids:
            self._last_seen_ts.pop(tid, None)
            self._prev_pos.pop(tid, None)
            self._origin_pos.pop(tid, None)
            self._prev_ts.pop(tid, None)
            self._first_crossing.pop(tid, None)
            self._track_speeds_mph.pop(tid, None)
            self._speed_anchor.pop(tid, None)
            self._stationary_since.pop(tid, None)
            self._crossed_pairs = {pair for pair in self._crossed_pairs if pair[0] != tid}
            for pair in [p for p in self._pending_crossings if p[0] == tid]:
                self._pending_crossings.pop(pair, None)

        return increments


def _bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0.0 else 0.0


def _bbox_ios(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection over smaller bounding-box area."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    min_area = min(area_a, area_b)
    return inter / min_area if min_area > 0.0 else 0.0


def _bbox_center_norm(
    bbox: tuple[float, float, float, float], frame_w: int, frame_h: int
) -> tuple[float, float]:
    return (
        ((bbox[0] + bbox[2]) * 0.5) / max(frame_w, 1),
        ((bbox[1] + bbox[3]) * 0.5) / max(frame_h, 1),
    )


@dataclass
class _CanonicalTrack:
    canonical_id: int
    raw_id: int
    is_person: bool
    class_id: int
    class_name: str
    class_votes: dict[int, float]
    bbox_xyxy: tuple[float, float, float, float]
    anchor_center_norm: tuple[float, float]
    stationary_frames: int = 0
    is_stationary: bool = False
    hits: int = 1
    last_seen_ts: float = 0.0
    last_real_seen_ts: float = 0.0
    last_confidence: float = 0.0
    max_confidence: float = 0.0


class TrackStabilizer:
    """Eliminates static car ID flickering, duplicate boxes, and class flips post-ByteTrack."""

    def __init__(self) -> None:
        self._next_id: int = 1
        self._canonical_tracks: dict[int, _CanonicalTrack] = {}
        self._raw_to_canonical: dict[int, int] = {}

    def reset(self) -> None:
        self._next_id = 1
        self._canonical_tracks.clear()
        self._raw_to_canonical.clear()

    def clear_raw_bindings(self) -> None:
        """Drop raw ByteTrack ID bindings when swapping detectors while keeping canonical tracks."""
        self._raw_to_canonical.clear()

    def stabilize(
        self,
        raw_tracks: list[TrackedDetection],
        frame_w: int,
        frame_h: int,
        now_ts: float,
        conf_threshold: float = 0.35,
    ) -> list[TrackedDetection]:
        def _sort_key(d: TrackedDetection) -> tuple[int, int, float]:
            cid = self._raw_to_canonical.get(d.track_id)
            ct = self._canonical_tracks.get(cid) if cid is not None else None
            return (1 if ct is not None else 0, ct.hits if ct is not None else 0, d.confidence)

        # 1. Suppress intra-frame duplicates (e.g. overlapping car + truck boxes on one SUV)
        sorted_raw = sorted(raw_tracks, key=_sort_key, reverse=True)
        deduped: list[TrackedDetection] = []
        for det in sorted_raw:
            is_dup = False
            for kept in deduped:
                if kept.is_person != det.is_person:
                    continue
                if _bbox_iou(kept.bbox_xyxy, det.bbox_xyxy) >= 0.55 or (
                    not det.is_person and _bbox_ios(kept.bbox_xyxy, det.bbox_xyxy) >= 0.82
                ):
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(det)

        # 2. Match existing raw track IDs when spatially consistent
        claimed_cids: set[int] = set()
        matched_pairs: list[tuple[TrackedDetection, _CanonicalTrack]] = []
        unmatched_dets: list[TrackedDetection] = []

        for det in deduped:
            cid = self._raw_to_canonical.get(det.track_id)
            ct = self._canonical_tracks.get(cid) if cid is not None else None
            if ct is not None and ct.is_person == det.is_person and ct.canonical_id not in claimed_cids:
                iou = _bbox_iou(det.bbox_xyxy, ct.bbox_xyxy)
                dcx, dcy = _bbox_center_norm(det.bbox_xyxy, frame_w, frame_h)
                ccx, ccy = _bbox_center_norm(ct.bbox_xyxy, frame_w, frame_h)
                c_dist = math.hypot(dcx - ccx, dcy - ccy)
                if iou >= 0.05 or c_dist <= 0.25:
                    claimed_cids.add(ct.canonical_id)
                    matched_pairs.append((det, ct))
                    continue
            unmatched_dets.append(det)

        # 3. Spatial ID stitching for stationary/briefly lost objects that got a new raw ID
        still_unmatched: list[TrackedDetection] = []
        for det in unmatched_dets:
            best_ct: _CanonicalTrack | None = None
            best_score = -1e9
            det_area = max(
                1.0, (det.bbox_xyxy[2] - det.bbox_xyxy[0]) * (det.bbox_xyxy[3] - det.bbox_xyxy[1])
            )
            dcx, dcy = _bbox_center_norm(det.bbox_xyxy, frame_w, frame_h)

            for ct in self._canonical_tracks.values():
                if ct.canonical_id in claimed_cids or ct.is_person != det.is_person:
                    continue
                max_age = 180.0 if (ct.is_stationary and not ct.is_person) else (15.0 if not ct.is_person else 8.0)
                if (now_ts - ct.last_real_seen_ts) > max_age:
                    continue

                iou = _bbox_iou(det.bbox_xyxy, ct.bbox_xyxy)
                ccx, ccy = _bbox_center_norm(ct.bbox_xyxy, frame_w, frame_h)
                c_dist = math.hypot(dcx - ccx, dcy - ccy)
                ct_area = max(
                    1.0, (ct.bbox_xyxy[2] - ct.bbox_xyxy[0]) * (ct.bbox_xyxy[3] - ct.bbox_xyxy[1])
                )
                size_ratio = min(det_area, ct_area) / max(det_area, ct_area)

                if ct.is_stationary and not ct.is_person:
                    ok = iou >= 0.40 or (c_dist <= 0.045 and size_ratio >= 0.45)
                else:
                    ok = iou >= 0.50 or (c_dist <= 0.040 and size_ratio >= 0.55)

                if ok:
                    score = iou - c_dist
                    if score > best_score:
                        best_score = score
                        best_ct = ct

            if best_ct is not None:
                claimed_cids.add(best_ct.canonical_id)
                self._raw_to_canonical[det.track_id] = best_ct.canonical_id
                best_ct.raw_id = det.track_id
                matched_pairs.append((det, best_ct))
            else:
                still_unmatched.append(det)

        # 4. Confidence gate, majority class voting, and adaptive EMA bbox smoothing
        low_keep = max(0.12, conf_threshold * 0.45)
        stabilized: list[TrackedDetection] = []

        for det, ct in matched_pairs:
            if ct.max_confidence < conf_threshold or det.confidence < low_keep:
                continue
            ct.hits += 1
            ct.last_seen_ts = now_ts
            ct.last_real_seen_ts = now_ts
            ct.last_confidence = det.confidence
            ct.max_confidence = max(ct.max_confidence, det.confidence)
            ct.raw_id = det.track_id

            if not ct.is_person:
                for k in list(ct.class_votes.keys()):
                    ct.class_votes[k] *= 0.98
                ct.class_votes[det.class_id] = ct.class_votes.get(det.class_id, 0.0) + det.confidence
                best_cls_id = max(ct.class_votes.items(), key=lambda kv: kv[1])[0]
                ct.class_id = best_cls_id
                ct.class_name = VEHICLE_CLASSES.get(best_cls_id, "vehicle")

            dcx, dcy = _bbox_center_norm(det.bbox_xyxy, frame_w, frame_h)
            ccx, ccy = _bbox_center_norm(ct.bbox_xyxy, frame_w, frame_h)
            shift = math.hypot(dcx - ccx, dcy - ccy)
            if shift <= 0.018:
                alpha = 0.25
            elif shift <= 0.045:
                alpha = 0.60
            else:
                alpha = 1.0

            ox1, oy1, ox2, oy2 = ct.bbox_xyxy
            nx1, ny1, nx2, ny2 = det.bbox_xyxy
            ct.bbox_xyxy = (
                (1.0 - alpha) * ox1 + alpha * nx1,
                (1.0 - alpha) * oy1 + alpha * ny1,
                (1.0 - alpha) * ox2 + alpha * nx2,
                (1.0 - alpha) * oy2 + alpha * ny2,
            )

            scx, scy = _bbox_center_norm(ct.bbox_xyxy, frame_w, frame_h)
            anchor_dist = math.hypot(scx - ct.anchor_center_norm[0], scy - ct.anchor_center_norm[1])
            if anchor_dist <= 0.025:
                ct.stationary_frames += 1
                if ct.stationary_frames >= 4:
                    ct.is_stationary = True
            elif anchor_dist > 0.045:
                ct.anchor_center_norm = (scx, scy)
                ct.stationary_frames = 0
                ct.is_stationary = False

            stabilized.append(
                TrackedDetection(
                    track_id=ct.canonical_id,
                    class_id=ct.class_id,
                    class_name=ct.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=ct.bbox_xyxy,
                    is_emergency=det.is_emergency,
                )
            )

        for det in still_unmatched:
            if det.confidence < conf_threshold:
                continue
            cid = self._next_id
            self._next_id += 1
            scx, scy = _bbox_center_norm(det.bbox_xyxy, frame_w, frame_h)
            votes = {} if det.is_person else {det.class_id: det.confidence}
            ct = _CanonicalTrack(
                canonical_id=cid,
                raw_id=det.track_id,
                is_person=det.is_person,
                class_id=det.class_id,
                class_name=det.class_name,
                class_votes=votes,
                bbox_xyxy=det.bbox_xyxy,
                anchor_center_norm=(scx, scy),
                stationary_frames=1,
                is_stationary=False,
                hits=1,
                last_seen_ts=now_ts,
                last_real_seen_ts=now_ts,
                last_confidence=det.confidence,
                max_confidence=det.confidence,
            )
            self._canonical_tracks[cid] = ct
            self._raw_to_canonical[det.track_id] = cid
            claimed_cids.add(cid)
            stabilized.append(
                TrackedDetection(
                    track_id=cid,
                    class_id=ct.class_id,
                    class_name=ct.class_name,
                    confidence=det.confidence,
                    bbox_xyxy=ct.bbox_xyxy,
                    is_emergency=det.is_emergency,
                )
            )

        # 5. Short coasting (anti-blink) for stationary vehicles missing for <= 2s
        for ct in self._canonical_tracks.values():
            if ct.canonical_id in claimed_cids:
                continue
            if (
                not ct.is_person
                and ct.is_stationary
                and ct.hits >= 5
                and ct.max_confidence >= conf_threshold
                and 0.0 < (now_ts - ct.last_real_seen_ts) <= 2.0
            ):
                if any(_bbox_iou(ct.bbox_xyxy, em.bbox_xyxy) >= 0.45 for em in stabilized):
                    continue
                stabilized.append(
                    TrackedDetection(
                        track_id=ct.canonical_id,
                        class_id=ct.class_id,
                        class_name=ct.class_name,
                        confidence=ct.last_confidence,
                        bbox_xyxy=ct.bbox_xyxy,
                    )
                )

        # 6. Prune expired canonical tracks & stale raw->canonical bindings
        for cid, ct in list(self._canonical_tracks.items()):
            max_ret = 180.0 if (ct.is_stationary and not ct.is_person) else 20.0
            if (now_ts - ct.last_real_seen_ts) > max_ret:
                del self._canonical_tracks[cid]

        for rid, cid in list(self._raw_to_canonical.items()):
            if cid not in self._canonical_tracks:
                del self._raw_to_canonical[rid]

        return stabilized


DFINE_MODEL_REPOS: dict[str, str] = {
    "dfine_n.onnx": "onnx-community/dfine_n_coco-ONNX",
    "dfine_s.onnx": "onnx-community/dfine_s_coco-ONNX",
}
_TARGET_CLASS_SET = frozenset(TARGET_CLASS_IDS)


def is_dfine_model(model_name: str) -> bool:
    """Return True if `model_name` refers to a D-FINE transformer ONNX model."""
    return Path(str(model_name).strip()).name.lower() in DFINE_MODEL_REPOS


def compute_optimal_input_hw(frame_w: int, frame_h: int, max_imgsz: int) -> tuple[int, int]:
    """Compute stream-aspect-matched (in_h, in_w) aligned to stride 32 without upscaling."""

    def _stride32(val: float) -> int:
        return max(32, int(math.ceil(val / 32.0)) * 32)

    cap = max(32, int(max_imgsz))
    if frame_w <= 0 or frame_h <= 0:
        side = _stride32(cap)
        return (side, side)

    scale = min(1.0, float(cap) / float(max(frame_w, frame_h)))
    in_w = _stride32(frame_w * scale)
    in_h = _stride32(frame_h * scale)
    return (in_h, in_w)


def _download_url_atomic(url: str, dest: Path, timeout: int = 120) -> None:
    """Download `url` to `dest` atomically via a sibling `.tmp` file."""
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(f".{dest.name}.tmp")
    try:
        with requests.get(url, stream=True, timeout=timeout) as resp:
            resp.raise_for_status()
            with open(tmp_path, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        fh.write(chunk)
        os.replace(tmp_path, dest)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _ensure_dfine_onnx(model_name: str, models_dir: Path | None) -> tuple[Path, dict[str, Any]]:
    """Ensure local cached D-FINE `model.onnx`, any external data file, and preprocessor config exist."""
    direct = Path(model_name)
    if direct.is_file() and direct.stat().st_size > 0:
        preproc: dict[str, Any] = {}
        preproc_sibling = direct.with_name("preprocessor_config.json")
        if preproc_sibling.is_file():
            try:
                preproc = json.loads(preproc_sibling.read_text(encoding="utf-8"))
            except Exception:
                preproc = {}
        return direct, preproc

    key = Path(str(model_name).strip()).name.lower()
    repo = DFINE_MODEL_REPOS.get(key)
    if repo is None:
        raise FileNotFoundError(f"Unknown D-FINE model '{model_name}'")

    cache_root = Path(models_dir) if models_dir is not None else Path("data/models")
    model_dir = cache_root / Path(key).stem
    model_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = model_dir / "model.onnx"
    preproc_path = model_dir / "preprocessor_config.json"

    if onnx_path.exists() and onnx_path.stat().st_size == 0:
        onnx_path.unlink(missing_ok=True)

    base_url = f"https://huggingface.co/{repo}/resolve/main"
    if not onnx_path.exists():
        _download_url_atomic(f"{base_url}/onnx/model.onnx", onnx_path, timeout=180)
        try:
            import onnx

            proto = onnx.load(str(onnx_path), load_external_data=False)
            ext_locations: set[str] = set()
            for tensor in proto.graph.initializer:
                if tensor.data_location == onnx.TensorProto.EXTERNAL:
                    for entry in tensor.external_data:
                        if entry.key == "location" and entry.value:
                            ext_locations.add(Path(entry.value).name)
            for ext_name in sorted(ext_locations):
                ext_path = model_dir / ext_name
                if not (ext_path.exists() and ext_path.stat().st_size > 0):
                    _download_url_atomic(f"{base_url}/onnx/{ext_name}", ext_path, timeout=180)
        except ImportError:
            pass
        except Exception:
            onnx_path.unlink(missing_ok=True)
            raise

    if not preproc_path.exists():
        try:
            _download_url_atomic(f"{base_url}/preprocessor_config.json", preproc_path, timeout=30)
        except Exception:
            preproc_path.unlink(missing_ok=True)

    preproc_cfg: dict[str, Any] = {}
    if preproc_path.is_file():
        try:
            preproc_cfg = json.loads(preproc_path.read_text(encoding="utf-8"))
        except Exception:
            preproc_cfg = {}
    return onnx_path, preproc_cfg


def _parse_tracker_yaml_args(tracker_yaml: str) -> types.SimpleNamespace:
    """Parse key-value ByteTrack YAML parameters into a SimpleNamespace for BYTETracker."""
    parsed: dict[str, Any] = {
        "tracker_type": "bytetrack",
        "track_high_thresh": 0.35,
        "track_low_thresh": 0.12,
        "new_track_thresh": 0.40,
        "track_buffer": 150,
        "match_thresh": 0.85,
        "fuse_score": False,
    }
    try:
        for raw_line in Path(tracker_yaml).read_text(encoding="utf-8").splitlines():
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            k, v = line.split(":", 1)
            key = k.strip()
            val = v.strip()
            low = val.lower()
            if low == "true":
                parsed[key] = True
            elif low == "false":
                parsed[key] = False
            else:
                try:
                    parsed[key] = int(val)
                except ValueError:
                    try:
                        parsed[key] = float(val)
                    except ValueError:
                        parsed[key] = val.strip("'\"")
    except OSError:
        pass
    return types.SimpleNamespace(**parsed)


_OV_GPU_PROBE_CACHE: tuple[Any, bool] | None = None


def _has_openvino_gpu() -> bool:
    """Return True if OpenVINO detects an Intel GPU device, cached per Core class."""
    global _OV_GPU_PROBE_CACHE
    try:
        import openvino as ov
    except ImportError:
        return False

    core_cls = getattr(ov, "Core", None)
    if core_cls is None:
        return False
    if _OV_GPU_PROBE_CACHE is not None and _OV_GPU_PROBE_CACHE[0] is core_cls:
        return _OV_GPU_PROBE_CACHE[1]

    has_gpu = False
    try:
        devs = list(core_cls().available_devices)
        has_gpu = any(d == "GPU" or d.startswith("GPU.") for d in devs)
    except Exception:
        has_gpu = False
    _OV_GPU_PROBE_CACHE = (core_cls, has_gpu)
    return has_gpu


def _configure_openvino_cache(core: Any, cache_root: Path) -> Path:
    """Enable persistent OpenCL (NEO) and OpenVINO GPU shader caches on disk."""
    neo_dir = cache_root / "neo_cache"
    ov_cache_dir = cache_root / "ov_cache"
    for d in (neo_dir, ov_cache_dir):
        try:
            d.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
    os.environ.setdefault("NEO_CACHE_PERSISTENT", "1")
    os.environ.setdefault("NEO_CACHE_DIR", str(neo_dir))
    if core is not None and hasattr(core, "set_property"):
        try:
            core.set_property({"CACHE_DIR": str(ov_cache_dir)})
        except Exception:
            pass
    return ov_cache_dir


class _DFineRunner:
    """NMS-free D-FINE ONNX/OpenVINO runner paired with Ultralytics BYTETracker."""

    def __init__(
        self,
        model_name: str,
        req_backend: str,
        models_dir: Path | None,
        tracker_yaml: str,
        threads: int = 4,
    ) -> None:
        from ultralytics.trackers.byte_tracker import BYTETracker

        self.onnx_path, self.preproc_cfg = _ensure_dfine_onnx(model_name, models_dir)
        self._tracker_args = _parse_tracker_yaml_args(tracker_yaml)
        self.tracker = BYTETracker(args=self._tracker_args)
        self.fallback_note = ""
        self._ort_session: Any = None
        self._ov_compiled: Any = None
        self._input_specs: list[tuple[str, tuple[Any, ...], str]] = []
        cache_root = Path(models_dir) if models_dir is not None else Path("data/models")

        compiled = False
        if req_backend in ("openvino", "openvino_gpu") or (
            req_backend == "auto" and _has_openvino_gpu()
        ):
            try:
                import openvino as ov

                core = ov.Core()
                _configure_openvino_cache(core, cache_root)
                if req_backend in ("openvino_gpu", "auto"):
                    devices = list(core.available_devices)
                    if any(d == "GPU" or d.startswith("GPU.") for d in devices):
                        try:
                            self._ov_compiled = core.compile_model(
                                str(self.onnx_path),
                                "GPU",
                                {
                                    "PERFORMANCE_HINT": "LATENCY",
                                    "INFERENCE_PRECISION_HINT": "f16",
                                },
                            )
                        except TypeError:
                            self._ov_compiled = core.compile_model(str(self.onnx_path), "GPU")
                        self.backend = "openvino_gpu"
                        self.device = "intel:gpu"
                        for inp in self._ov_compiled.inputs:
                            pshape = tuple(
                                int(dim.get_length()) if dim.is_static else -1
                                for dim in inp.get_partial_shape()
                            )
                            self._input_specs.append(
                                (inp.any_name, pshape, str(inp.get_element_type()))
                            )
                        compiled = True
                    elif req_backend == "openvino_gpu":
                        diag = collect_hardware_diagnostics(run_clinfo=False)
                        self.fallback_note = (
                            diag["short_fallback_reason"]
                            or "Intel GPU unavailable; fell back to ONNX"
                        )
                else:
                    self._ov_compiled = core.compile_model(str(self.onnx_path), "CPU")
                    self.backend = "openvino"
                    self.device = "cpu"
                    for inp in self._ov_compiled.inputs:
                        pshape = tuple(
                            int(dim.get_length()) if dim.is_static else -1
                            for dim in inp.get_partial_shape()
                        )
                        self._input_specs.append(
                            (inp.any_name, pshape, str(inp.get_element_type()))
                        )
                    compiled = True
            except Exception as exc:
                if req_backend == "openvino_gpu":
                    diag = collect_hardware_diagnostics(run_clinfo=False)
                    self.fallback_note = (
                        diag["short_fallback_reason"]
                        or "Intel GPU unavailable; fell back to ONNX"
                    )
                elif req_backend == "openvino":
                    self.fallback_note = f"OpenVINO unavailable ({exc}); fell back to ONNX"

        if not compiled:
            import onnxruntime as ort

            opts = ort.SessionOptions()
            opts.intra_op_num_threads = threads
            opts.inter_op_num_threads = 1
            opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
            opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
            self._ort_session = ort.InferenceSession(
                str(self.onnx_path),
                sess_options=opts,
                providers=["CPUExecutionProvider"],
            )
            self.backend = "onnx"
            self.device = "cpu"
            if req_backend == "pytorch" and not self.fallback_note:
                self.fallback_note = "D-FINE runs via ONNX Runtime"
            for inp in self._ort_session.get_inputs():
                self._input_specs.append((inp.name, tuple(inp.shape), str(inp.type)))

        self.input_hw = self._resolve_input_hw()

    def _resolve_input_hw(self) -> tuple[int, int]:
        size_cfg = self.preproc_cfg.get("size")
        if isinstance(size_cfg, dict):
            sh = int(size_cfg.get("height", 0) or 0)
            sw = int(size_cfg.get("width", 0) or 0)
            if sh > 0 and sw > 0:
                return (sh, sw)
        for _, shape, _ in self._input_specs:
            if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
                if shape[2] > 0 and shape[3] > 0:
                    return (int(shape[2]), int(shape[3]))
        return (640, 640)

    def reset(self) -> None:
        self.tracker.reset()

    @staticmethod
    def _decode_dfine_outputs(
        outputs: list[Any], frame_w: int, frame_h: int, low_conf: float
    ) -> np.ndarray:
        """Decode D-FINE ONNX outputs into (N, 6) [x1, y1, x2, y2, score, cls_id] float32 array."""
        arrays = [np.asarray(out) for out in outputs]
        if not arrays:
            return np.empty((0, 6), dtype=np.float32)

        boxes_arr: np.ndarray | None = None
        logits_arr: np.ndarray | None = None
        labels_arr: np.ndarray | None = None
        scores_arr: np.ndarray | None = None

        for arr in arrays:
            if arr.ndim >= 2 and arr.shape[-1] == 4 and boxes_arr is None:
                boxes_arr = arr.astype(np.float32)
            elif arr.ndim >= 2 and arr.shape[-1] >= 5 and logits_arr is None:
                logits_arr = arr.astype(np.float32)

        if logits_arr is not None and boxes_arr is not None:
            logits2d = logits_arr[0] if logits_arr.ndim == 3 else logits_arr
            boxes2d = boxes_arr[0] if boxes_arr.ndim == 3 else boxes_arr
            if logits2d.size == 0 or boxes2d.size == 0:
                return np.empty((0, 6), dtype=np.float32)
            if float(logits2d.min()) < 0.0 or float(logits2d.max()) > 1.0:
                probs = 1.0 / (1.0 + np.exp(-np.clip(logits2d, -50.0, 50.0)))
            else:
                probs = logits2d
            if probs.shape[-1] == 81:
                probs = probs[..., :80]
            cls_ids = np.argmax(probs, axis=-1).astype(np.int32)
            scores = probs[np.arange(probs.shape[0]), cls_ids].astype(np.float32)
        elif len(arrays) >= 3 and boxes_arr is not None:
            # Fallback for 3-output (labels, boxes, scores) exports
            for arr in arrays:
                if arr is boxes_arr:
                    continue
                flat = arr.reshape(-1)
                if np.issubdtype(flat.dtype, np.integer) and labels_arr is None:
                    labels_arr = flat.astype(np.int32)
                elif scores_arr is None:
                    scores_arr = flat.astype(np.float32)
            if labels_arr is None or scores_arr is None:
                return np.empty((0, 6), dtype=np.float32)
            boxes2d = boxes_arr.reshape(-1, 4)
            cls_ids = labels_arr
            scores = scores_arr
        else:
            return np.empty((0, 6), dtype=np.float32)

        target_mask = np.isin(cls_ids, TARGET_CLASS_IDS)
        keep = (scores >= float(low_conf)) & target_mask
        if not np.any(keep):
            return np.empty((0, 6), dtype=np.float32)

        kept_boxes = boxes2d[keep]
        kept_scores = scores[keep]
        kept_cls = cls_ids[keep].astype(np.float32)

        if float(kept_boxes.max()) <= 1.5 and logits_arr is not None:
            cx = kept_boxes[:, 0] * float(frame_w)
            cy = kept_boxes[:, 1] * float(frame_h)
            bw = kept_boxes[:, 2] * float(frame_w)
            bh = kept_boxes[:, 3] * float(frame_h)
            x1 = cx - (bw * 0.5)
            y1 = cy - (bh * 0.5)
            x2 = cx + (bw * 0.5)
            y2 = cy + (bh * 0.5)
        elif float(kept_boxes.max()) <= 1.5:
            x1 = kept_boxes[:, 0] * float(frame_w)
            y1 = kept_boxes[:, 1] * float(frame_h)
            x2 = kept_boxes[:, 2] * float(frame_w)
            y2 = kept_boxes[:, 3] * float(frame_h)
        else:
            x1 = kept_boxes[:, 0]
            y1 = kept_boxes[:, 1]
            x2 = kept_boxes[:, 2]
            y2 = kept_boxes[:, 3]

        x1 = np.clip(x1, 0.0, float(frame_w))
        y1 = np.clip(y1, 0.0, float(frame_h))
        x2 = np.clip(x2, 0.0, float(frame_w))
        y2 = np.clip(y2, 0.0, float(frame_h))

        valid = (x2 > (x1 + 1.0)) & (y2 > (y1 + 1.0))
        if not np.any(valid):
            return np.empty((0, 6), dtype=np.float32)

        return np.column_stack(
            [x1[valid], y1[valid], x2[valid], y2[valid], kept_scores[valid], kept_cls[valid]]
        ).astype(np.float32)

    def infer_detections(self, frame: np.ndarray, low_conf: float) -> np.ndarray:
        frame_h, frame_w = frame.shape[:2]
        in_h, in_w = self.input_hw

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        if self.preproc_cfg.get("do_rescale", True):
            resized *= float(self.preproc_cfg.get("rescale_factor", 1.0 / 255.0))
        if self.preproc_cfg.get("do_normalize", False):
            mean = np.asarray(
                self.preproc_cfg.get("image_mean", [0.485, 0.456, 0.406]), dtype=np.float32
            ).reshape(1, 1, 3)
            std = np.asarray(
                self.preproc_cfg.get("image_std", [0.229, 0.224, 0.225]), dtype=np.float32
            ).reshape(1, 1, 3)
            resized = (resized - mean) / std
        pixel_values = np.ascontiguousarray(resized.transpose(2, 0, 1)[None, ...], dtype=np.float32)

        feed: dict[str, np.ndarray] = {}
        for idx, (inp_name, _shape, inp_type) in enumerate(self._input_specs):
            if idx == 0 or "pixel" in inp_name or "image" in inp_name:
                feed[inp_name] = pixel_values
            elif "size" in inp_name:
                dtype = np.int32 if "int32" in inp_type.lower() or "i32" in inp_type.lower() else np.int64
                feed[inp_name] = np.array([[frame_h, frame_w]], dtype=dtype)
        if not feed:
            feed = {"pixel_values": pixel_values}

        if self._ov_compiled is not None:
            res = self._ov_compiled(feed)
            outputs = [res[out] for out in self._ov_compiled.outputs]
        else:
            outputs = self._ort_session.run(None, feed)

        return self._decode_dfine_outputs(outputs, frame_w=frame_w, frame_h=frame_h, low_conf=low_conf)

    def track(
        self,
        frame: np.ndarray,
        low_conf: float,
        high_conf: float,
        low_track: float,
        new_thresh: float,
    ) -> list[TrackedDetection]:
        import torch
        from ultralytics.engine.results import Boxes

        frame_h, frame_w = frame.shape[:2]
        self._tracker_args.track_high_thresh = high_conf
        self._tracker_args.track_low_thresh = low_track
        self._tracker_args.new_track_thresh = new_thresh

        dets = self.infer_detections(frame, low_conf=low_conf)
        boxes_tensor = (
            torch.from_numpy(dets) if len(dets) > 0 else torch.empty((0, 6), dtype=torch.float32)
        )
        tracked = self.tracker.update(Boxes(boxes_tensor, orig_shape=(frame_h, frame_w)), img=frame)

        raw_output: list[TrackedDetection] = []
        if isinstance(tracked, np.ndarray) and tracked.ndim == 2 and tracked.shape[0] > 0:
            for row in tracked:
                x1, y1, x2, y2, tid, score, cls_id = row[:7]
                cid = int(cls_id)
                if cid not in _TARGET_CLASS_SET:
                    continue
                cls_name = "person" if cid == PERSON_CLASS_ID else VEHICLE_CLASSES.get(cid, "vehicle")
                raw_output.append(
                    TrackedDetection(
                        track_id=int(tid),
                        class_id=cid,
                        class_name=cls_name,
                        confidence=float(score),
                        bbox_xyxy=(float(x1), float(y1), float(x2), float(y2)),
                    )
                )
        return raw_output


class MotionGate:
    """Lightweight OpenCV motion gate that throttles neural-network inference on static scenes."""

    def __init__(
        self,
        width: int = 160,
        height: int = 90,
        diff_threshold: int = 14,
        pixel_ratio_threshold: float = 0.002,
        cooldown_seconds: float = 2.5,
    ) -> None:
        self.width = max(32, int(width))
        self.height = max(32, int(height))
        self.diff_threshold = max(1, int(diff_threshold))
        self.pixel_ratio_threshold = max(0.0001, float(pixel_ratio_threshold))
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self._prev_gray: np.ndarray | None = None
        self._last_infer_gray: np.ndarray | None = None
        self._last_infer_mono: float = 0.0
        self._motion_until_mono: float = 0.0
        self._excl_sig: Any = None
        self._valid_mask: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_gray = None
        self._last_infer_gray = None
        self._last_infer_mono = 0.0
        self._motion_until_mono = 0.0

    def _get_valid_mask(self, zones: list[ZoneModel] | None) -> np.ndarray | None:
        excl_polys = [
            tuple(
                (round(float(pt[0]), 4), round(float(pt[1]), 4))
                for pt in z.points
            )
            for z in (zones or [])
            if z.enabled and getattr(z, "zone_type", "red_zone") == "exclusion" and len(z.points) >= 3
        ]
        sig = tuple(excl_polys)
        if sig == self._excl_sig:
            return self._valid_mask
        self._excl_sig = sig
        if not excl_polys:
            self._valid_mask = None
            return None

        mask = np.full((self.height, self.width), 255, dtype=np.uint8)
        for poly in excl_polys:
            pts = np.array(
                [
                    [
                        max(0, min(self.width, int(round(px * self.width)))),
                        max(0, min(self.height, int(round(py * self.height)))),
                    ]
                    for px, py in poly
                ],
                dtype=np.int32,
            )
            cv2.fillPoly(mask, [pts], 0)
        self._valid_mask = mask
        return mask

    def should_infer(
        self,
        frame: np.ndarray,
        zones: list[ZoneModel] | None = None,
        now_mono: float | None = None,
        idle_interval: float = 2.0,
    ) -> tuple[bool, str]:
        """Return (should_run_nn_inference, motion_state['active' | 'idle'])."""
        now = time.monotonic() if now_mono is None else float(now_mono)
        small = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY) if small.ndim == 3 else small
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if (
            self._prev_gray is None
            or self._last_infer_gray is None
            or self._prev_gray.shape != gray.shape
        ):
            self._prev_gray = gray
            self._last_infer_gray = gray
            self._last_infer_mono = now
            self._motion_until_mono = now + self.cooldown_seconds
            return (True, "active")

        diff = np.maximum(
            cv2.absdiff(gray, self._prev_gray),
            cv2.absdiff(gray, self._last_infer_gray),
        )
        self._prev_gray = gray

        valid_mask = self._get_valid_mask(zones)
        if valid_mask is not None:
            diff = cv2.bitwise_and(diff, valid_mask)
            valid_pixels = max(1, int(np.count_nonzero(valid_mask)))
        else:
            valid_pixels = self.width * self.height

        _, motion_bin = cv2.threshold(diff, self.diff_threshold, 255, cv2.THRESH_BINARY)
        motion_bin = cv2.erode(motion_bin, np.ones((2, 2), dtype=np.uint8), iterations=1)
        changed_ratio = float(np.count_nonzero(motion_bin)) / float(valid_pixels)

        if changed_ratio >= self.pixel_ratio_threshold:
            self._motion_until_mono = now + self.cooldown_seconds
            self._last_infer_mono = now
            self._last_infer_gray = gray
            return (True, "active")

        if now < self._motion_until_mono:
            self._last_infer_mono = now
            self._last_infer_gray = gray
            return (True, "active")

        if (now - self._last_infer_mono) >= max(0.25, float(idle_interval)):
            self._last_infer_mono = now
            self._last_infer_gray = gray
            return (True, "idle")

        return (False, "idle")


def _resolve_model_source(model_name: str, models_dir: Path | None) -> str:
    """Resolve YOLO weights path so downloaded `.pt` files persist inside `models_dir`."""
    if models_dir is None or "/" in model_name or "\\" in model_name:
        return model_name
    models_dir.mkdir(parents=True, exist_ok=True)
    cached = models_dir / model_name
    if cached.exists():
        return str(cached)
    local = Path(model_name)
    if local.exists():
        return str(local)
    return str(cached)


def _ensure_tracker_yaml(models_dir: Path | None, conf: float = 0.35) -> str:
    """Write a stationary-friendly ByteTrack config YAML and return its path."""
    target_dir = Path(models_dir) if models_dir is not None else Path("/tmp")
    target_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = target_dir / "bytetrack_car_tracker.yaml"
    high_thresh = round(max(0.15, min(conf, 0.85)), 3)
    new_thresh = round(min(0.90, max(high_thresh + 0.05, 0.40)), 3)
    yaml_content = (
        "tracker_type: bytetrack\n"
        f"track_high_thresh: {high_thresh}\n"
        "track_low_thresh: 0.12\n"
        f"new_track_thresh: {new_thresh}\n"
        "track_buffer: 150\n"
        "match_thresh: 0.85\n"
        "fuse_score: False\n"
    )
    yaml_path.write_text(yaml_content, encoding="utf-8")
    return str(yaml_path)


def collect_hardware_diagnostics(
    cpuinfo_path: Path = Path("/proc/cpuinfo"),
    dmi_dir: Path = Path("/sys/class/dmi/id"),
    pci_devices_dir: Path = Path("/sys/bus/pci/devices"),
    dri_dir: Path = Path("/dev/dri"),
    opencl_vendors_dir: Path = Path("/etc/OpenCL/vendors"),
    os_release_path: Path = Path("/etc/os-release"),
    vaapi_driver_dirs: tuple[Path, ...] = (
        Path("/usr/lib/x86_64-linux-gnu/dri"),
        Path("/usr/lib/aarch64-linux-gnu/dri"),
        Path("/usr/lib/dri"),
    ),
    ffmpeg_bin: str | None = None,
    run_clinfo: bool = True,
) -> dict[str, Any]:
    """Inspect host CPU, virtualization, PCI GPUs, /dev/dri nodes, OpenCL ICDs, and ML runtimes."""
    os_release = platform.platform()
    if os_release_path.exists():
        try:
            for line in os_release_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if line.startswith("PRETTY_NAME="):
                    os_release = line.split("=", 1)[1].strip().strip('"')
                    break
        except OSError:
            pass

    model_name = ""
    arm_model = ""
    hypervisor_flag = False
    if cpuinfo_path.exists():
        try:
            for line in cpuinfo_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                if ":" not in line:
                    continue
                key_raw, val_raw = line.split(":", 1)
                key_stripped = key_raw.strip()
                key_lower = key_stripped.lower()
                if key_lower == "model name" and not model_name:
                    model_name = val_raw.strip()
                elif (key_stripped == "Model" or key_lower == "hardware") and not arm_model:
                    arm_model = val_raw.strip()
                elif key_lower == "flags":
                    if "hypervisor" in val_raw.split():
                        hypervisor_flag = True
        except OSError:
            pass
    cpu_model = model_name or arm_model or platform.processor() or platform.machine()
    cpu_cores = int(os.cpu_count() or 1)

    board_parts: list[str] = []
    if dmi_dir.exists():
        for fname in ("sys_vendor", "product_name"):
            fpath = dmi_dir / fname
            if fpath.exists():
                try:
                    val = fpath.read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
                    if val and val not in board_parts:
                        board_parts.append(val)
                except OSError:
                    pass
    board = " ".join(board_parts).strip()
    if not board:
        dt_model = Path("/proc/device-tree/model")
        if dt_model.exists():
            try:
                board = dt_model.read_text(encoding="utf-8", errors="ignore").replace("\x00", "").strip()
            except OSError:
                pass

    vendor_map = {
        "0x8086": "Intel",
        "0x1002": "AMD",
        "0x10de": "NVIDIA",
        "0x1af4": "VirtIO",
        "0x1234": "QEMU/Bochs",
        "0x15ad": "VMware",
    }
    known_devices = {
        "0x46d0": "Intel Alder Lake-N UHD Graphics",
        "0x46d1": "Intel Alder Lake-N UHD Graphics",
        "0x46d2": "Intel Alder Lake-N UHD Graphics",
        "0x46d3": "Intel Twin Lake UHD Graphics",
        "0x46d4": "Intel Twin Lake UHD Graphics",
        "0x4e55": "Intel Jasper Lake UHD Graphics",
        "0x4e61": "Intel Jasper Lake UHD Graphics",
        "0x4e71": "Intel Jasper Lake UHD Graphics",
        "0x9a49": "Intel Iris Xe Graphics",
        "0x46a6": "Intel Iris Xe Graphics",
        "0xa7a0": "Intel Iris Xe Graphics",
    }
    pci_gpus: list[dict[str, str]] = []
    if pci_devices_dir.exists():
        try:
            for dev_dir in sorted(pci_devices_dir.iterdir()):
                cls_path = dev_dir / "class"
                if not cls_path.exists():
                    continue
                try:
                    cls_hex = cls_path.read_text(encoding="utf-8", errors="ignore").strip().lower()
                except OSError:
                    continue
                if not cls_hex.startswith("0x03"):
                    continue
                vendor_id = ""
                device_id = ""
                driver = ""
                try:
                    vendor_id = (dev_dir / "vendor").read_text(encoding="utf-8", errors="ignore").strip().lower()
                except OSError:
                    pass
                try:
                    device_id = (dev_dir / "device").read_text(encoding="utf-8", errors="ignore").strip().lower()
                except OSError:
                    pass
                drv_path = dev_dir / "driver"
                try:
                    if drv_path.is_symlink():
                        driver = drv_path.resolve().name
                    elif drv_path.is_file():
                        driver = drv_path.read_text(encoding="utf-8", errors="ignore").strip()
                    elif drv_path.exists():
                        driver = drv_path.name
                except OSError:
                    pass
                vendor_name = vendor_map.get(vendor_id, vendor_id or "Unknown")
                pci_gpus.append(
                    {
                        "slot": dev_dir.name,
                        "vendor_id": vendor_id,
                        "vendor": vendor_name,
                        "device_id": device_id,
                        "description": known_devices.get(device_id, f"{vendor_name} GPU ({device_id})" if device_id else vendor_name),
                        "class_id": cls_hex,
                        "driver": driver,
                    }
                )
        except OSError:
            pass

    vm_tokens = ("qemu", "kvm", "virtualbox", "vmware", "proxmox", "standard pc", "bochs", "hyper-v")
    physical_vendors = {"Intel", "AMD", "NVIDIA"}
    virtualized = bool(
        hypervisor_flag
        or any(tok in board.lower() for tok in vm_tokens)
        or (pci_gpus and all(g["vendor"] not in physical_vendors for g in pci_gpus))
    )

    dri_devices: list[dict[str, Any]] = []
    if dri_dir.exists():
        try:
            for entry in sorted(dri_dir.iterdir()):
                try:
                    st = entry.stat()
                    dri_devices.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "readable": bool(os.access(entry, os.R_OK)),
                            "writable": bool(os.access(entry, os.W_OK)),
                            "gid": int(st.st_gid),
                            "mode": oct(st.st_mode & 0o777),
                        }
                    )
                except OSError:
                    dri_devices.append(
                        {
                            "name": entry.name,
                            "path": str(entry),
                            "readable": False,
                            "writable": False,
                            "gid": -1,
                            "mode": "unknown",
                        }
                    )
        except OSError:
            pass

    opencl_icds: list[str] = []
    if opencl_vendors_dir.exists():
        try:
            opencl_icds = sorted(f.name for f in opencl_vendors_dir.glob("*.icd"))
        except OSError:
            pass

    clinfo_summary: list[str] = []
    if run_clinfo and shutil.which("clinfo"):
        try:
            proc = subprocess.run(
                ["clinfo", "--list"],
                capture_output=True,
                text=True,
                timeout=3,
            )
            lines = [ln.strip() for ln in (proc.stdout or proc.stderr or "").splitlines() if ln.strip()]
            clinfo_summary = lines[:8]
        except Exception:
            pass

    ov_info: dict[str, Any] = {
        "installed": False,
        "available_devices": [],
        "device_names": {},
        "gpu_available": False,
    }
    try:
        import openvino as ov

        ov_info["installed"] = True
        core = ov.Core()
        devs = list(core.available_devices)
        ov_info["available_devices"] = devs
        dev_names: dict[str, str] = {}
        for d in devs:
            try:
                dev_names[d] = str(core.get_property(d, "FULL_DEVICE_NAME"))
            except Exception:
                dev_names[d] = d
        ov_info["device_names"] = dev_names
        ov_info["gpu_available"] = any(d == "GPU" or d.startswith("GPU.") for d in devs)
    except Exception:
        pass

    ort_info: dict[str, Any] = {"installed": False, "version": "", "providers": []}
    try:
        import onnxruntime as ort

        ort_info["installed"] = True
        ort_info["version"] = str(getattr(ort, "__version__", ""))
        ort_info["providers"] = list(ort.get_available_providers())
    except Exception:
        pass

    torch_info: dict[str, bool] = {"cuda_available": False, "mps_available": False}
    try:
        import torch

        torch_info["cuda_available"] = bool(torch.cuda.is_available())
        torch_info["mps_available"] = bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )
    except Exception:
        pass

    render_nodes = [d for d in dri_devices if d["name"].startswith("renderD")]
    render_rw = any(d["readable"] and d["writable"] for d in render_nodes)
    va_driver = ""
    for drv_dir in vaapi_driver_dirs:
        for drv_name in ("iHD_drv_video.so", "i965_drv_video.so"):
            if (drv_dir / drv_name).is_file():
                va_driver = drv_name
                break
        if va_driver:
            break
    resolved_ffmpeg = shutil.which("ffmpeg") if ffmpeg_bin is None else ffmpeg_bin
    vaapi_info: dict[str, Any] = {
        "driver_installed": bool(va_driver),
        "driver": va_driver,
        "ffmpeg_installed": bool(resolved_ffmpeg),
        "ready": bool(va_driver and resolved_ffmpeg and render_rw),
    }

    has_linux_probe = any(
        p.exists() for p in (cpuinfo_path, pci_devices_dir, dri_dir, opencl_vendors_dir)
    )
    gpu_available = False
    gpu_name = ""
    diagnosis = ""
    short_fallback_reason = ""

    if ov_info["gpu_available"]:
        gpu_dev = next(
            (d for d in ov_info["available_devices"] if d == "GPU" or d.startswith("GPU.")),
            "GPU",
        )
        gpu_name = ov_info["device_names"].get(gpu_dev) or "Intel GPU"
        gpu_available = True
        diagnosis = f"Intel GPU ready via OpenVINO ({gpu_name})"
        if vaapi_info["ready"]:
            diagnosis += " • QuickSync VA-API decode ready"
    elif torch_info["cuda_available"]:
        gpu_available = True
        gpu_name = "NVIDIA CUDA GPU"
        diagnosis = "NVIDIA CUDA GPU available"
        short_fallback_reason = "OpenVINO requires Intel GPU; CUDA available"
    elif torch_info["mps_available"] and not has_linux_probe:
        gpu_available = True
        gpu_name = "Apple Metal (MPS)"
        diagnosis = "Apple Metal (MPS) GPU available (OpenVINO Intel GPU requires Linux x86_64)"
        short_fallback_reason = "Intel GPU unavailable on macOS; fell back to ONNX"
    else:
        vendors = {g["vendor"] for g in pci_gpus}
        if pci_gpus:
            gpu_name = pci_gpus[0]["description"]
        if "AMD" in vendors and "Intel" not in vendors:
            diagnosis = "AMD GPU detected; OpenVINO GPU only supports Intel GPUs — use ONNX Runtime CPU."
            short_fallback_reason = "AMD GPU detected (OpenVINO requires Intel); fell back to ONNX"
        elif not render_nodes and virtualized:
            diagnosis = (
                "Virtualized host has no /dev/dri/renderD* node — pass the host PCIe iGPU "
                "through to the Home Assistant VM."
            )
            short_fallback_reason = "No /dev/dri/renderD* in VM (enable iGPU passthrough); fell back to ONNX"
        elif not render_nodes:
            diagnosis = (
                "No /dev/dri/renderD* GPU render node exposed — verify host iGPU is enabled in BIOS "
                "and Add-on Protection Mode allows /dev/dri."
            )
            short_fallback_reason = "No /dev/dri/renderD* device exposed; fell back to ONNX"
        elif not render_rw:
            diagnosis = (
                "/dev/dri/renderD* exists but lacks read/write permission — disable Protection Mode "
                "on the Add-on Info page."
            )
            short_fallback_reason = "/dev/dri/renderD* permission denied (check Protection Mode); fell back to ONNX"
        elif not opencl_icds:
            diagnosis = (
                "/dev/dri/renderD* is accessible, but no OpenCL ICD is installed in "
                "/etc/OpenCL/vendors — rebuild/update the add-on container."
            )
            short_fallback_reason = "Missing Intel OpenCL ICD in container; fell back to ONNX"
        elif not ov_info["installed"]:
            diagnosis = "OpenVINO runtime is not installed on this platform/architecture."
            short_fallback_reason = "OpenVINO not installed; fell back to ONNX"
        else:
            diagnosis = (
                f"Render node and OpenCL ICD ({', '.join(opencl_icds)}) exist, but OpenVINO only found "
                f"{ov_info['available_devices']} — check host kernel i915/xe driver or Protection Mode."
            )
            short_fallback_reason = "Intel GPU unavailable in OpenVINO; fell back to ONNX"

    return {
        "platform": sys.platform,
        "arch": platform.machine(),
        "kernel": platform.release(),
        "os_release": os_release,
        "cpu_model": cpu_model,
        "cpu_cores": cpu_cores,
        "board": board,
        "virtualized": virtualized,
        "pci_gpus": pci_gpus,
        "dri_devices": dri_devices,
        "opencl_icds": opencl_icds,
        "clinfo_summary": clinfo_summary,
        "openvino": ov_info,
        "onnxruntime": ort_info,
        "pytorch": torch_info,
        "vaapi": vaapi_info,
        "gpu_available": gpu_available,
        "gpu_name": gpu_name,
        "diagnosis": diagnosis,
        "short_fallback_reason": short_fallback_reason,
    }


class ObjectDetector:
    """Ultralytics YOLO / D-FINE + ByteTrack wrapper with ONNX Runtime & OpenVINO acceleration."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        imgsz: int = 960,
        inference_backend: str = "auto",
        models_dir: Path | None = None,
        input_hw: tuple[int, int] | None = None,
    ) -> None:
        import torch
        from ultralytics import YOLO

        req_backend = (inference_backend or "auto").strip().lower()
        self.model_name = model_name
        self.imgsz = imgsz
        self.requested_backend = req_backend
        self.fallback_note = ""
        self.backend = "pytorch"
        self.device = "cpu"
        self.stabilizer = TrackStabilizer()
        self._tracker_yaml = _ensure_tracker_yaml(models_dir, 0.35)
        self._is_dfine = is_dfine_model(model_name)
        self._dfine_runner: _DFineRunner | None = None
        threads = min(4, max(1, os.cpu_count() or 4))

        def _configure_cpu_threads() -> None:
            os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
            os.environ.setdefault("KMP_BLOCKTIME", "0")
            try:
                torch.set_num_threads(threads)
            except RuntimeError:
                pass
            cv2.setNumThreads(min(2, threads))

        if self._is_dfine:
            _configure_cpu_threads()
            runner = _DFineRunner(
                model_name=model_name,
                req_backend=req_backend,
                models_dir=models_dir,
                tracker_yaml=self._tracker_yaml,
                threads=threads,
            )
            self._dfine_runner = runner
            self.backend = runner.backend
            self.device = runner.device
            self.fallback_note = runner.fallback_note
            self.input_hw: tuple[int, int] = runner.input_hw
            self.model: Any = None
            return

        if input_hw is not None and input_hw[0] > 0 and input_hw[1] > 0:
            self.input_hw = (int(input_hw[0]), int(input_hw[1]))
        else:
            side = max(32, int(imgsz))
            self.input_hw = (side, side)
        target_size: int | tuple[int, int] = (
            self.input_hw[0] if self.input_hw[0] == self.input_hw[1] else self.input_hw
        )

        fallback_reason: str | None = None
        if req_backend == "onnx":
            _configure_cpu_threads()
            onnx_model = self._try_load_onnx(YOLO, model_name, target_size, models_dir, threads)
            if onnx_model is not None:
                self.model = onnx_model
                self.device = "cpu"
                self.backend = "onnx"
                return
            fallback_reason = "ONNX unavailable"
        elif req_backend == "openvino_gpu":
            _configure_cpu_threads()
            ov_gpu_model = self._try_load_openvino(
                YOLO, model_name, target_size, models_dir, require_gpu=True
            )
            if ov_gpu_model is not None:
                self.model = ov_gpu_model
                self.device = "intel:gpu"
                self.backend = "openvino_gpu"
                return
            # Never silently drop from openvino_gpu onto OpenVINO CPU spin-wait; prefer ONNX CPU instead.
            onnx_model = self._try_load_onnx(YOLO, model_name, target_size, models_dir, threads)
            if onnx_model is not None:
                self.model = onnx_model
                self.device = "cpu"
                self.backend = "onnx"
                diag = collect_hardware_diagnostics(run_clinfo=False)
                self.fallback_note = (
                    diag["short_fallback_reason"]
                    or "Intel GPU unavailable; fell back to ONNX"
                )
                return
            fallback_reason = "OpenVINO GPU & ONNX unavailable"
        elif req_backend == "openvino":
            _configure_cpu_threads()
            ov_model = self._try_load_openvino(
                YOLO, model_name, target_size, models_dir, require_gpu=False
            )
            if ov_model is not None:
                self.model = ov_model
                self.device = "cpu"
                self.backend = "openvino"
                return
            fallback_reason = "OpenVINO unavailable"

        if torch.backends.mps.is_available():
            self.device = "mps"
            self.backend = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.backend = "cuda"
        elif req_backend == "auto" and _has_openvino_gpu():
            _configure_cpu_threads()
            ov_auto_model = self._try_load_openvino(
                YOLO, model_name, target_size, models_dir, require_gpu=True
            )
            if ov_auto_model is not None:
                self.model = ov_auto_model
                self.device = "intel:gpu"
                self.backend = "openvino_gpu"
                return
            self.device = "cpu"
            self.backend = "pytorch"
        else:
            self.device = "cpu"
            self.backend = "pytorch"
            _configure_cpu_threads()

        if fallback_reason:
            self.fallback_note = f"{fallback_reason}; fell back to {self.backend.upper()}"

        self.model = YOLO(_resolve_model_source(model_name, models_dir))

    @property
    def input_resolution_label(self) -> str:
        """Return effective model input geometry formatted as `WxH`."""
        return f"{self.input_hw[1]}x{self.input_hw[0]}"

    def reset(self) -> None:
        """Reset post-tracker stabilization and ByteTrack state."""
        self.stabilizer.reset()
        if self._dfine_runner is not None:
            self._dfine_runner.reset()
            return
        predictor = getattr(self.model, "predictor", None)
        for tr in getattr(predictor, "trackers", None) or []:
            if hasattr(tr, "reset"):
                tr.reset()

    @staticmethod
    def _try_load_onnx(
        yolo_cls: Any,
        model_name: str,
        imgsz: int | tuple[int, int],
        models_dir: Path | None,
        threads: int = 4,
    ) -> Any | None:
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort
        except ImportError:
            return None

        in_h, in_w = (int(imgsz), int(imgsz)) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))
        export_imgsz: int | tuple[int, int] = in_h if in_h == in_w else (in_h, in_w)
        size_tag = f"imgsz{in_h}" if in_h == in_w else f"{in_h}x{in_w}"

        cache_root = Path(models_dir) if models_dir is not None else Path("data/models")
        cache_root.mkdir(parents=True, exist_ok=True)
        onnx_path = cache_root / f"{Path(model_name).stem}_{size_tag}.onnx"

        try:
            if onnx_path.exists() and onnx_path.stat().st_size == 0:
                onnx_path.unlink(missing_ok=True)
            if not (onnx_path.exists() and onnx_path.stat().st_size > 0):
                pt_model = yolo_cls(_resolve_model_source(model_name, models_dir))
                exported_path = Path(
                    str(pt_model.export(format="onnx", imgsz=export_imgsz, half=False, simplify=False))
                )
                if exported_path.exists() and exported_path.resolve() != onnx_path.resolve():
                    shutil.move(str(exported_path), str(onnx_path))
            if onnx_path.exists() and onnx_path.stat().st_size > 0:
                y_model = yolo_cls(str(onnx_path), task="detect")

                def _tune_ort_session(predictor: Any) -> None:
                    if getattr(predictor, "_ort_tuned", False):
                        return
                    backend = getattr(predictor.model, "backend", None)
                    if backend is not None and hasattr(backend, "session"):
                        opts = ort.SessionOptions()
                        opts.intra_op_num_threads = threads
                        opts.inter_op_num_threads = 1
                        opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
                        opts.add_session_config_entry("session.inter_op.allow_spinning", "0")
                        backend.session = ort.InferenceSession(
                            str(onnx_path),
                            sess_options=opts,
                            providers=["CPUExecutionProvider"],
                        )
                        predictor._ort_tuned = True

                y_model.add_callback("on_predict_start", _tune_ort_session)
                return y_model
        except Exception as exc:
            onnx_path.unlink(missing_ok=True)
            print(f"[detector] ONNX export/load fallback to PyTorch: {exc}", flush=True)
        return None

    @staticmethod
    def _try_load_openvino(
        yolo_cls: Any,
        model_name: str,
        imgsz: int | tuple[int, int],
        models_dir: Path | None,
        require_gpu: bool = False,
    ) -> Any | None:
        try:
            import openvino as ov
        except ImportError:
            return None

        cache_root = Path(models_dir) if models_dir is not None else Path("data/models")
        cache_root.mkdir(parents=True, exist_ok=True)

        if require_gpu:
            try:
                core = ov.Core()
                _configure_openvino_cache(core, cache_root)
                devices = list(core.available_devices)
                if not any(d == "GPU" or d.startswith("GPU.") for d in devices):
                    print(
                        f"[detector] Intel iGPU not detected in OpenVINO devices ({devices}); falling back from openvino_gpu",
                        flush=True,
                    )
                    return None
            except Exception as exc:
                print(f"[detector] OpenVINO device probe failed: {exc}", flush=True)
                return None
        else:
            try:
                _configure_openvino_cache(ov.Core(), cache_root)
            except Exception:
                _configure_openvino_cache(None, cache_root)

        in_h, in_w = (int(imgsz), int(imgsz)) if isinstance(imgsz, int) else (int(imgsz[0]), int(imgsz[1]))
        export_imgsz: int | tuple[int, int] = in_h if in_h == in_w else (in_h, in_w)
        size_tag = f"imgsz{in_h}" if in_h == in_w else f"{in_h}x{in_w}"
        use_fp16 = bool(require_gpu)
        prec_tag = "_fp16" if use_fp16 else ""

        ov_dir = cache_root / f"{Path(model_name).stem}_{size_tag}{prec_tag}_openvino_model"

        try:
            if not (ov_dir.exists() and any(ov_dir.glob("*.xml"))):
                pt_model = yolo_cls(_resolve_model_source(model_name, models_dir))
                exported_path = Path(
                    str(pt_model.export(format="openvino", imgsz=export_imgsz, half=use_fp16))
                )
                if exported_path.exists() and exported_path.resolve() != ov_dir.resolve():
                    if ov_dir.exists():
                        shutil.rmtree(ov_dir)
                    shutil.move(str(exported_path), str(ov_dir))
            if ov_dir.exists() and any(ov_dir.glob("*.xml")):
                return yolo_cls(str(ov_dir), task="detect")
        except Exception as exc:
            print(f"[detector] OpenVINO export/load fallback: {exc}", flush=True)
        return None

    def detect_and_track(
        self,
        frame: np.ndarray,
        imgsz: int = 960,
        conf: float = 0.35,
        now_ts: float | None = None,
    ) -> list[TrackedDetection]:
        ts = time.time() if now_ts is None else now_ts
        frame_h, frame_w = frame.shape[:2]
        low_conf = round(max(0.10, min(0.15, conf * 0.45)), 3)
        high_conf = round(max(0.15, min(conf, 0.90)), 3)
        new_thresh = round(min(0.90, max(high_conf + 0.05, high_conf)), 3)
        low_track = min(0.12, round(low_conf * 0.9, 3))

        if self._dfine_runner is not None:
            raw_dfine = self._dfine_runner.track(
                frame=frame,
                low_conf=low_conf,
                high_conf=high_conf,
                low_track=low_track,
                new_thresh=new_thresh,
            )
            return self.stabilizer.stabilize(
                raw_tracks=raw_dfine,
                frame_w=frame_w,
                frame_h=frame_h,
                now_ts=ts,
                conf_threshold=conf,
            )

        predictor = getattr(self.model, "predictor", None)
        for tr in getattr(predictor, "trackers", None) or []:
            tr_args = getattr(tr, "args", None)
            if tr_args is not None:
                tr_args.track_high_thresh = high_conf
                tr_args.track_low_thresh = low_track
                tr_args.new_track_thresh = new_thresh

        if self.backend in ("onnx", "openvino", "openvino_gpu"):
            run_imgsz: int | tuple[int, int] = (
                self.input_hw[0] if self.input_hw[0] == self.input_hw[1] else self.input_hw
            )
        else:
            if frame_w >= 160 and frame_h >= 160:
                dyn_hw = compute_optimal_input_hw(frame_w, frame_h, imgsz)
            else:
                side = max(32, int(imgsz))
                dyn_hw = (side, side)
            self.input_hw = dyn_hw
            run_imgsz = dyn_hw[0] if dyn_hw[0] == dyn_hw[1] else dyn_hw

        results = self.model.track(
            frame,
            persist=True,
            tracker=self._tracker_yaml,
            imgsz=run_imgsz,
            conf=low_conf,
            iou=0.55,
            agnostic_nms=True,
            classes=TARGET_CLASS_IDS,
            device=self.device,
            verbose=False,
        )
        raw_output: list[TrackedDetection] = []
        if results:
            boxes = results[0].boxes
            if boxes is not None and boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                clss = boxes.cls.int().cpu().tolist()
                confs = boxes.conf.float().cpu().tolist()
                xyxys = boxes.xyxy.float().cpu().tolist()

                for tid, cls_id, score, xyxy in zip(ids, clss, confs, xyxys):
                    x1, y1, x2, y2 = [float(v) for v in xyxy]
                    cls_name = "person" if cls_id == PERSON_CLASS_ID else VEHICLE_CLASSES.get(cls_id, "vehicle")
                    raw_output.append(
                        TrackedDetection(
                            track_id=int(tid),
                            class_id=int(cls_id),
                            class_name=cls_name,
                            confidence=float(score),
                            bbox_xyxy=(x1, y1, x2, y2),
                        )
                    )

        return self.stabilizer.stabilize(
            raw_tracks=raw_output,
            frame_w=frame_w,
            frame_h=frame_h,
            now_ts=ts,
            conf_threshold=conf,
        )


def _hex_to_bgr(hex_color: str) -> tuple[int, int, int]:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return (68, 68, 239)
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (b, g, r)


def format_dwell_duration(seconds: float) -> str:
    """Format parked dwell seconds cleanly as '45s', '2m 15s', or '1h 5m 12s'."""
    total = max(0, int(round(seconds)))
    if total < 60:
        return f"{total}s"
    mins, secs = divmod(total, 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hours, mins = divmod(mins, 60)
    return f"{hours}h {mins}m {secs}s"


def _draw_dashed_line(
    img: np.ndarray,
    pt1: tuple[int, int],
    pt2: tuple[int, int],
    color: tuple[int, int, int],
    thickness: int,
    dash_px: int = 12,
    gap_px: int = 8,
) -> None:
    """Draw a dashed anti-aliased line segment between two pixel points."""
    x1, y1 = pt1
    x2, y2 = pt2
    dist = math.hypot(x2 - x1, y2 - y1)
    if dist < 1.0:
        cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)
        return
    ux = (x2 - x1) / dist
    uy = (y2 - y1) / dist
    pos = 0.0
    step = max(2.0, float(dash_px + gap_px))
    while pos < dist:
        end_pos = min(dist, pos + dash_px)
        p_a = (int(round(x1 + ux * pos)), int(round(y1 + uy * pos)))
        p_b = (int(round(x1 + ux * end_pos)), int(round(y1 + uy * end_pos)))
        cv2.line(img, p_a, p_b, color, thickness, cv2.LINE_AA)
        pos += step


def annotate_frame(
    frame: np.ndarray,
    tracks: list[TrackedDetection],
    track_statuses: dict[int, dict[str, Any]],
    zones: list[ZoneModel],
    dwell_threshold: float,
    hotspot: HotspotTracker | None = None,
    show_heatmap: bool = False,
    counting_lines: list[CountingLineModel] | None = None,
    annotation_scale: float = 1.0,
    person_dwell_threshold: float = 2.0,
) -> np.ndarray:
    """Render red zones, optional heatmap overlay, counting lines, and tracked bounding boxes."""
    out = frame.copy()
    h, w = out.shape[:2]
    scale = (w / 1280.0) * max(0.4, min(annotation_scale, 3.0))
    box_thick = max(1, int(round(2.0 * scale)))
    font_scale = max(0.36, 0.58 * scale)
    text_thick = max(1, int(round(1.8 * scale)))
    pad_x = max(4, int(round(6 * scale)))
    pad_y = max(4, int(round(6 * scale)))
    dot_r = max(3, int(round(4 * scale)))

    if show_heatmap and hotspot is not None:
        out = hotspot.render_overlay(out, alpha=0.45)

    # Draw red-zone & exclusion-mask polygons
    overlay = out.copy()
    for zone in zones:
        if not zone.enabled or len(zone.points) < 3:
            continue
        pts = np.array([[int(p[0] * w), int(p[1] * h)] for p in zone.points], dtype=np.int32)
        bgr = (180, 163, 148) if zone.zone_type == "exclusion" else _hex_to_bgr(zone.color)
        cv2.fillPoly(overlay, [pts], bgr)
        cv2.polylines(out, [pts], isClosed=True, color=bgr, thickness=box_thick)
        label_pt = (int(pts[0][0]), max(int(24 * scale), int(pts[0][1]) - pad_y))
        if zone.zone_type == "exclusion":
            z_label = f"MASK: {zone.name}"
        elif zone.zone_type == "person_zone":
            z_label = f"PERSON: {zone.name}"
        else:
            z_label = zone.name
        cv2.putText(out, z_label, label_pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr, text_thick, cv2.LINE_AA)

    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)

    # Draw traffic counting & speed measurement lines
    for ln in counting_lines or []:
        if not ln.enabled:
            continue
        pt1 = (int(ln.p1[0] * w), int(ln.p1[1] * h))
        pt2 = (int(ln.p2[0] * w), int(ln.p2[1] * h))
        role = getattr(ln, "line_role", "both")
        line_thick = max(2, box_thick + 1)
        mid = ((pt1[0] + pt2[0]) // 2, max(int(26 * scale), (pt1[1] + pt2[1]) // 2 - pad_y))
        if role == "speed":
            lcolor = (
                _hex_to_bgr("#f59e0b")
                if ln.color.lower() == "#38bdf8"
                else _hex_to_bgr(ln.color)
            )
            _draw_dashed_line(out, pt1, pt2, lcolor, line_thick)
            ln_label = f"SPEED: {ln.name}"
        else:
            lcolor = _hex_to_bgr(ln.color)
            cv2.line(out, pt1, pt2, lcolor, line_thick, cv2.LINE_AA)
            total_ln = ln.count_a_to_b + ln.count_b_to_a
            ln_label = f"{ln.name}: {total_ln} ({ln.count_a_to_b}/{ln.count_b_to_a})"
        cv2.circle(out, pt1, dot_r + 1, lcolor, -1)
        cv2.circle(out, pt2, dot_r + 1, lcolor, -1)
        cv2.putText(out, ln_label, mid, cv2.FONT_HERSHEY_SIMPLEX, font_scale, lcolor, text_thick, cv2.LINE_AA)

    # Draw tracked detections
    for det in tracks:
        x1, y1, x2, y2 = [int(v) for v in det.bbox_xyxy]
        st = track_statuses.get(det.track_id, {})
        if st.get("emergency"):
            color = (0, 215, 255)  # Gold / Amber
            label = f"EMERGENCY (EXEMPT) #{det.track_id} {det.class_name}"
        elif st.get("violation"):
            color = (48, 48, 255)  # Bright Red
            dur_label = format_dwell_duration(st.get("dwell_seconds", 0.0))
            label = f"ILLEGAL PARK #{det.track_id} {det.class_name} ({dur_label})"
        elif st.get("in_red_zone"):
            color = (0, 165, 255)  # Orange
            label = f"#{det.track_id} {det.class_name} {st.get('dwell_seconds', 0):.0f}/{dwell_threshold:.0f}s"
        elif det.is_person and st.get("person_violation"):
            color = (247, 85, 168)  # Violet/Magenta (#a855f7)
            dur_label = format_dwell_duration(st.get("person_dwell_seconds", 0.0))
            label = f"PERSON ALERT #{det.track_id} ({dur_label})"
        elif det.is_person and st.get("in_person_zone"):
            color = (250, 165, 192)  # Soft violet/lavender
            label = f"#{det.track_id} person {st.get('person_dwell_seconds', 0):.0f}/{person_dwell_threshold:.0f}s"
        elif det.is_person:
            color = (235, 206, 135)  # Light cyan/blue
            label = f"#{det.track_id} person"
        else:
            color = (113, 204, 46)  # Emerald green
            label = f"#{det.track_id} {det.class_name}"
        if det.is_vehicle and st.get("speed_mph") is not None:
            label = f"{label} | {st['speed_mph']:.0f} mph"
        if det.is_vehicle and st.get("license_plate"):
            label = f"{label} [{st['license_plate']}]"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, box_thick)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thick)
        bg_top = max(0, y1 - th - pad_y * 2)
        cv2.rectangle(out, (x1, bg_top), (x1 + tw + pad_x * 2, y1), color, -1)
        cv2.putText(
            out,
            label,
            (x1 + pad_x, max(th + pad_y, y1 - pad_y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (15, 15, 15),
            text_thick,
            cv2.LINE_AA,
        )

        # Draw bottom-center anchor dot for vehicles or watched people
        if det.is_vehicle or st.get("in_person_zone") or st.get("person_violation"):
            cx = int((x1 + x2) * 0.5)
            cv2.circle(out, (cx, y2), dot_r, color, -1)

    return out


def compute_ground_transforms(
    calib: GroundCalibrationModel, out_w: int = 800, out_h: int = 600
) -> dict[str, Any]:
    """Compute 3x3 homography matrices between normalized camera coords, ground (ft), and BEV."""
    src_norm = np.array(calib.points, dtype=np.float32)
    w_ft = float(calib.width_ft)
    l_ft = float(calib.length_ft)
    dst_ft = np.array([[0.0, 0.0], [w_ft, 0.0], [w_ft, l_ft], [0.0, l_ft]], dtype=np.float32)

    img_norm_to_ground = cv2.getPerspectiveTransform(src_norm, dst_ft)
    ground_to_img_norm = cv2.getPerspectiveTransform(dst_ft, src_norm)

    # Place calibrated rectangle centered in BEV canvas with room around it for street context
    px_per_ft = min((out_w * 0.42) / max(w_ft, 1e-3), (out_h * 0.42) / max(l_ft, 1e-3))
    ox = (out_w - w_ft * px_per_ft) * 0.5
    oy = (out_h - l_ft * px_per_ft) * 0.5
    bev_norm_rect = np.array(
        [
            [ox / out_w, oy / out_h],
            [(ox + w_ft * px_per_ft) / out_w, oy / out_h],
            [(ox + w_ft * px_per_ft) / out_w, (oy + l_ft * px_per_ft) / out_h],
            [ox / out_w, (oy + l_ft * px_per_ft) / out_h],
        ],
        dtype=np.float32,
    )
    img_norm_to_bev_norm = cv2.getPerspectiveTransform(src_norm, bev_norm_rect)
    bev_norm_to_img_norm = cv2.getPerspectiveTransform(bev_norm_rect, src_norm)

    return {
        "img_norm_to_ground": img_norm_to_ground,
        "ground_to_img_norm": ground_to_img_norm,
        "img_norm_to_bev_norm": img_norm_to_bev_norm,
        "bev_norm_to_img_norm": bev_norm_to_img_norm,
        "px_per_ft": px_per_ft,
        "origin_px": (ox, oy),
        "out_w": out_w,
        "out_h": out_h,
    }


def project_points(pts: list[list[float]], matrix: np.ndarray) -> list[list[float]]:
    """Project a list of [x, y] points through a 3x3 perspective matrix."""
    if not pts:
        return []
    arr = np.array([pts], dtype=np.float32)
    warped = cv2.perspectiveTransform(arr, matrix)[0]
    return [[round(float(x), 4), round(float(y), 4)] for x, y in warped]


def render_birdseye_jpeg(
    frame: np.ndarray,
    calib: GroundCalibrationModel,
    zones: list[ZoneModel],
    lines: list[CountingLineModel],
    out_w: int = 800,
    out_h: int = 600,
) -> bytes | None:
    """Warp the camera frame into a top-down bird's-eye view with grid, zones, and lines."""
    h, w = frame.shape[:2]
    tf = compute_ground_transforms(calib, out_w=out_w, out_h=out_h)

    scale_in = np.diag([1.0 / max(w, 1), 1.0 / max(h, 1), 1.0])
    scale_out = np.diag([float(out_w), float(out_h), 1.0])
    h_px = scale_out @ tf["img_norm_to_bev_norm"] @ scale_in

    bev = cv2.warpPerspective(
        frame,
        h_px,
        (out_w, out_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(15, 23, 42),
    )

    # Draw ground grid lines in feet
    px_per_ft = tf["px_per_ft"]
    ox, oy = tf["origin_px"]
    step_ft = max(float(calib.grid_step_ft), 1.0)
    step_px = step_ft * px_per_ft
    if step_px >= 8.0:
        x = ox % step_px
        while x < out_w:
            cv2.line(bev, (int(round(x)), 0), (int(round(x)), out_h), (71, 85, 105), 1)
            x += step_px
        y = oy % step_px
        while y < out_h:
            cv2.line(bev, (0, int(round(y))), (out_w, int(round(y))), (71, 85, 105), 1)
            y += step_px

    # Draw calibrated reference rectangle
    rect_pts = np.array(
        [
            [int(ox), int(oy)],
            [int(ox + calib.width_ft * px_per_ft), int(oy)],
            [int(ox + calib.width_ft * px_per_ft), int(oy + calib.length_ft * px_per_ft)],
            [int(ox), int(oy + calib.length_ft * px_per_ft)],
        ],
        dtype=np.int32,
    )
    cv2.polylines(bev, [rect_pts], isClosed=True, color=(34, 211, 238), thickness=2)

    # Project existing zones onto top-down map
    for z in zones:
        if not z.enabled or len(z.points) < 3:
            continue
        bev_norm = project_points(z.points, tf["img_norm_to_bev_norm"])
        pts = np.array([[int(p[0] * out_w), int(p[1] * out_h)] for p in bev_norm], dtype=np.int32)
        bgr = (180, 163, 148) if z.zone_type == "exclusion" else _hex_to_bgr(z.color)
        cv2.polylines(bev, [pts], isClosed=True, color=bgr, thickness=2)

    # Project existing traffic lines onto top-down map
    for ln in lines:
        if not ln.enabled:
            continue
        bev_norm = project_points([ln.p1, ln.p2], tf["img_norm_to_bev_norm"])
        p1 = (int(bev_norm[0][0] * out_w), int(bev_norm[0][1] * out_h))
        p2 = (int(bev_norm[1][0] * out_w), int(bev_norm[1][1] * out_h))
        line_bgr = (11, 158, 245) if getattr(ln, "line_role", "both") == "speed" else (248, 189, 56)
        cv2.line(bev, p1, p2, line_bgr, 2)

    cv2.putText(
        bev,
        f"Bird's-Eye View (Grid: {step_ft:g} ft | Ref: {calib.width_ft:g}x{calib.length_ft:g} ft)",
        (12, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (226, 232, 240),
        1,
        cv2.LINE_AA,
    )
    ok, buf = cv2.imencode(".jpg", bev, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def detect_red_curb_zones(
    frame: np.ndarray,
    calib: GroundCalibrationModel | None = None,
    lane_width_ft: float = 7.5,
) -> dict[str, Any]:
    """Detect painted red curbs in a frame and project them into parking-lane polygons."""
    if frame is None or frame.size == 0:
        return {"detected": False, "used_ground_calibration": calib is not None, "zones": [], "message": "Empty frame."}

    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, 65, 45]), np.array([12, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([166, 65, 45]), np.array([179, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)
    mask[: int(h * 0.12), :] = 0

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = max(80.0, float(w * h) * 0.0004)
    min_len = 0.07 * min(w, h)

    candidates: list[tuple[float, float, np.ndarray]] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < min_area:
            continue
        _, (rw, rh), _ = cv2.minAreaRect(cnt)
        long_side = max(rw, rh)
        short_side = max(min(rw, rh), 1.0)
        if long_side < min_len or (long_side / short_side) < 2.0:
            continue
        candidates.append((area, short_side, cnt))

    if not candidates:
        return {
            "detected": False,
            "used_ground_calibration": calib is not None,
            "zones": [],
            "message": "No painted red curb detected in the current frame (works best in daylight with an unobstructed curb).",
        }

    candidates.sort(key=lambda item: item[0], reverse=True)
    best_area = candidates[0][0]
    selected = [c for c in candidates if c[0] >= best_area * 0.3][:2]

    tf = compute_ground_transforms(calib) if calib is not None else None
    zones: list[dict[str, Any]] = []

    for idx, (_, short_side, cnt) in enumerate(selected, start=1):
        eps = 0.015 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True)
        if len(approx) < 2:
            continue
        curb_norm = np.array([[float(p[0][0]) / w, float(p[0][1]) / h] for p in approx], dtype=np.float32)

        if tf is not None and calib is not None:
            g_pts = np.array(project_points(curb_norm.tolist(), tf["img_norm_to_ground"]), dtype=np.float32)
            vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(g_pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()]
            dir_v = np.array([vx, vy], dtype=np.float32)
            origin = np.array([x0, y0], dtype=np.float32)
            proj = (g_pts - origin) @ dir_v
            g_a = origin + float(proj.min()) * dir_v
            g_b = origin + float(proj.max()) * dir_v
            normal = np.array([-vy, vx], dtype=np.float32)
            g_mid = 0.5 * (g_a + g_b)
            road_ref = np.array([float(calib.width_ft) * 0.5, float(calib.length_ft) * 0.5], dtype=np.float32)
            if float(np.dot(road_ref - g_mid, normal)) < 0.0:
                normal = -normal
            ground_poly = [
                (g_a - 0.5 * normal).tolist(),
                (g_b - 0.5 * normal).tolist(),
                (g_b + float(lane_width_ft) * normal).tolist(),
                (g_a + float(lane_width_ft) * normal).tolist(),
            ]
            img_poly = project_points(ground_poly, tf["ground_to_img_norm"])
        else:
            vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(curb_norm, cv2.DIST_L2, 0, 0.01, 0.01).flatten()]
            dir_v = np.array([vx, vy], dtype=np.float32)
            origin = np.array([x0, y0], dtype=np.float32)
            proj = (curb_norm - origin) @ dir_v
            p_a = origin + float(proj.min()) * dir_v
            p_b = origin + float(proj.max()) * dir_v
            normal = np.array([-vy, vx], dtype=np.float32)
            mid = 0.5 * (p_a + p_b)
            road_ref = np.array([0.5, 0.72], dtype=np.float32)
            if float(np.dot(road_ref - mid, normal)) < 0.0:
                normal = -normal
            offset = max(0.06, min(0.12, (short_side / float(max(h, 1))) * 3.0))
            img_poly = [
                (p_a - 0.01 * normal).tolist(),
                (p_b - 0.01 * normal).tolist(),
                (p_b + offset * normal).tolist(),
                (p_a + offset * normal).tolist(),
            ]

        clamped = [[round(max(0.0, min(1.0, float(px))), 4), round(max(0.0, min(1.0, float(py))), 4)] for px, py in img_poly]
        zones.append(
            {
                "name": f"Auto Red Zone {idx}",
                "points": clamped,
                "color": "#ef4444",
                "zone_type": "red_zone",
            }
        )

    return {
        "detected": len(zones) > 0,
        "used_ground_calibration": tf is not None,
        "zones": zones,
        "message": (
            f"Detected {len(zones)} red curb zone(s)"
            + (" using 3D ground lane offset." if tf is not None else " (approximate 2D offset; calibrate 3D Ground Grid for perspective-accurate width).")
        ),
    }
