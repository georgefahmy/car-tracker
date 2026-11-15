from __future__ import annotations

import math
import os
import platform
import shutil
import time
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
class MonitorStepResult:
    """Output of a single frame update in RedZoneMonitor."""

    active_cars: int = 0
    active_people: int = 0
    new_unique_cars: int = 0
    new_total_cars: int = 0
    new_unique_people: int = 0
    new_total_people: int = 0
    new_violations: list[ViolationEvent] = field(default_factory=list)
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

    def reset(self) -> None:
        self._seen_frames.clear()
        self._counted_tracks.clear()
        self._last_seen_ts.clear()
        self._initial_pos.clear()
        self._counted_vehicle_pos.clear()
        self._dwell_states.clear()
        self._flagged_pairs.clear()

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
    ) -> MonitorStepResult:
        res = MonitorStepResult()
        enabled_zones = [
            z for z in zones if z.enabled and z.zone_type == "red_zone" and len(z.points) >= 3
        ]
        active_pairs: set[tuple[int, str]] = set()
        active_vehicle_ids = {d.track_id for d in tracks if d.is_vehicle}

        for det in tracks:
            tid = det.track_id
            self._last_seen_ts[tid] = now_ts
            self._seen_frames[tid] = self._seen_frames.get(tid, 0) + 1
            nx, ny = det.bottom_center_norm(frame_w, frame_h) if det.is_vehicle else (0.0, 0.0)
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
            }

            if det.is_vehicle and enabled_zones and not (exempt_emergency_vehicles and det.is_emergency):
                for zone in enabled_zones:
                    if not point_in_normalized_polygon(nx, ny, zone.points):
                        continue

                    pair = (tid, zone.id)
                    active_pairs.add(pair)
                    state = self._dwell_states.get(pair)
                    if state is None:
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
                        if dist <= movement_threshold:
                            state.stationary_seconds += dt
                        else:
                            # Vehicle is actively moving; reset stationary timer
                            state.anchor_pos = (nx, ny)
                            state.stationary_seconds = 0.0
                            state.entered_at = now_ts
                        state.last_seen_at = now_ts

                    if state.stationary_seconds >= dwell_seconds and not state.flagged:
                        state.flagged = True
                        self._flagged_pairs.add(pair)
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

                    status["in_red_zone"] = True
                    status["zone_name"] = zone.name
                    status["dwell_seconds"] = max(status["dwell_seconds"], round(state.stationary_seconds, 1))
                    status["violation"] = status["violation"] or state.flagged

            res.track_statuses[tid] = status

        # Clear dwell states only after exceeding the dropout grace period
        for pair in list(self._dwell_states.keys()):
            if (
                pair not in active_pairs
                and (now_ts - self._dwell_states[pair].last_seen_at) > dwell_grace_seconds
            ):
                del self._dwell_states[pair]

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
    ) -> None:
        self._crossed_pairs.add((tid, ln.id))
        a_inc, b_inc = increments.get(ln.id, (0, 0))
        if side_prev < 0:
            increments[ln.id] = (a_inc + 1, b_inc)
        else:
            increments[ln.id] = (a_inc, b_inc + 1)

        first = self._first_crossing.get(tid)
        if first is None:
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
                            tid, det, ln, pend_ts, pend_pt, pend_side_prev, tf, now_ts, curr, increments
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
                        tid, det, ln, cross_ts, cross_pt, side_prev, tf, now_ts, curr, increments
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


class ObjectDetector:
    """Ultralytics YOLO + built-in ByteTrack wrapper with ONNX Runtime & OpenVINO acceleration."""

    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        imgsz: int = 960,
        inference_backend: str = "auto",
        models_dir: Path | None = None,
    ) -> None:
        import torch
        from ultralytics import YOLO

        self.model_name = model_name
        self.imgsz = imgsz
        self.backend = "pytorch"
        self.stabilizer = TrackStabilizer()
        self._tracker_yaml = _ensure_tracker_yaml(models_dir, 0.35)
        threads = min(4, max(1, os.cpu_count() or 4))

        if torch.backends.mps.is_available():
            self.device = "mps"
            self.backend = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
            self.backend = "cuda"
        else:
            self.device = "cpu"
            os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")
            os.environ.setdefault("KMP_BLOCKTIME", "0")
            torch.set_num_threads(threads)
            cv2.setNumThreads(min(2, threads))

        if self.device == "cpu":
            if inference_backend == "onnx":
                onnx_model = self._try_load_onnx(YOLO, model_name, imgsz, models_dir, threads)
                if onnx_model is not None:
                    self.model = onnx_model
                    self.backend = "onnx"
                    return
            elif inference_backend == "openvino_gpu":
                ov_gpu_model = self._try_load_openvino(YOLO, model_name, imgsz, models_dir, require_gpu=True)
                if ov_gpu_model is not None:
                    self.model = ov_gpu_model
                    self.device = "intel:gpu"
                    self.backend = "openvino_gpu"
                    return
                # Never silently drop from openvino_gpu onto OpenVINO CPU spin-wait; prefer ONNX CPU instead.
                onnx_model = self._try_load_onnx(YOLO, model_name, imgsz, models_dir, threads)
                if onnx_model is not None:
                    self.model = onnx_model
                    self.backend = "onnx"
                    return
            elif inference_backend == "openvino":
                ov_model = self._try_load_openvino(YOLO, model_name, imgsz, models_dir, require_gpu=False)
                if ov_model is not None:
                    self.model = ov_model
                    self.backend = "openvino"
                    return

        self.model = YOLO(_resolve_model_source(model_name, models_dir))

    def reset(self) -> None:
        """Reset post-tracker stabilization and Ultralytics ByteTrack state."""
        self.stabilizer.reset()
        predictor = getattr(self.model, "predictor", None)
        for tr in getattr(predictor, "trackers", None) or []:
            if hasattr(tr, "reset"):
                tr.reset()

    @staticmethod
    def _try_load_onnx(
        yolo_cls: Any,
        model_name: str,
        imgsz: int,
        models_dir: Path | None,
        threads: int = 4,
    ) -> Any | None:
        try:
            import onnx  # noqa: F401
            import onnxruntime as ort
        except ImportError:
            return None

        cache_root = Path(models_dir) if models_dir is not None else Path("data/models")
        cache_root.mkdir(parents=True, exist_ok=True)
        onnx_path = cache_root / f"{Path(model_name).stem}_imgsz{imgsz}.onnx"

        try:
            if not onnx_path.exists():
                pt_model = yolo_cls(_resolve_model_source(model_name, models_dir))
                exported_path = Path(
                    str(pt_model.export(format="onnx", imgsz=imgsz, half=False, simplify=False))
                )
                if exported_path.exists() and exported_path.resolve() != onnx_path.resolve():
                    shutil.move(str(exported_path), str(onnx_path))
            if onnx_path.exists():
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
            print(f"[detector] ONNX export/load fallback to PyTorch: {exc}", flush=True)
        return None

    @staticmethod
    def _try_load_openvino(
        yolo_cls: Any,
        model_name: str,
        imgsz: int,
        models_dir: Path | None,
        require_gpu: bool = False,
    ) -> Any | None:
        try:
            import openvino as ov
        except ImportError:
            return None

        if require_gpu:
            try:
                core = ov.Core()
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

        cache_root = Path(models_dir) if models_dir is not None else Path("data/models")
        cache_root.mkdir(parents=True, exist_ok=True)
        ov_dir = cache_root / f"{Path(model_name).stem}_imgsz{imgsz}_openvino_model"

        try:
            if not (ov_dir.exists() and any(ov_dir.glob("*.xml"))):
                pt_model = yolo_cls(_resolve_model_source(model_name, models_dir))
                exported_path = Path(str(pt_model.export(format="openvino", imgsz=imgsz, half=False)))
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

        predictor = getattr(self.model, "predictor", None)
        for tr in getattr(predictor, "trackers", None) or []:
            tr_args = getattr(tr, "args", None)
            if tr_args is not None:
                tr_args.track_high_thresh = high_conf
                tr_args.track_low_thresh = low_track
                tr_args.new_track_thresh = new_thresh

        results = self.model.track(
            frame,
            persist=True,
            tracker=self._tracker_yaml,
            imgsz=imgsz,
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
        z_label = f"MASK: {zone.name}" if zone.zone_type == "exclusion" else zone.name
        cv2.putText(out, z_label, label_pt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr, text_thick, cv2.LINE_AA)

    out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)

    # Draw traffic counting lines
    for ln in counting_lines or []:
        if not ln.enabled:
            continue
        pt1 = (int(ln.p1[0] * w), int(ln.p1[1] * h))
        pt2 = (int(ln.p2[0] * w), int(ln.p2[1] * h))
        lcolor = _hex_to_bgr(ln.color)
        cv2.line(out, pt1, pt2, lcolor, max(2, box_thick + 1), cv2.LINE_AA)
        cv2.circle(out, pt1, dot_r + 1, lcolor, -1)
        cv2.circle(out, pt2, dot_r + 1, lcolor, -1)
        mid = ((pt1[0] + pt2[0]) // 2, max(int(26 * scale), (pt1[1] + pt2[1]) // 2 - pad_y))
        total_ln = ln.count_a_to_b + ln.count_b_to_a
        ln_label = f"{ln.name}: {total_ln} ({ln.count_a_to_b}/{ln.count_b_to_a})"
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
            label = f"ILLEGAL PARK #{det.track_id} {det.class_name} ({st.get('dwell_seconds', 0):.0f}s)"
        elif st.get("in_red_zone"):
            color = (0, 165, 255)  # Orange
            label = f"#{det.track_id} {det.class_name} {st.get('dwell_seconds', 0):.0f}/{dwell_threshold:.0f}s"
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

        # Draw bottom-center anchor dot for vehicles
        if det.is_vehicle:
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
        cv2.line(bev, p1, p2, (248, 189, 56), 2)

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
