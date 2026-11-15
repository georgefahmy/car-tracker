from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
        self._dwell_states: dict[tuple[int, str], DwellState] = {}
        self._flagged_pairs: set[tuple[int, str]] = set()

    def reset(self) -> None:
        self._seen_frames.clear()
        self._counted_tracks.clear()
        self._last_seen_ts.clear()
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
    ) -> MonitorStepResult:
        res = MonitorStepResult()
        enabled_zones = [
            z for z in zones if z.enabled and z.zone_type == "red_zone" and len(z.points) >= 3
        ]
        active_pairs: set[tuple[int, str]] = set()

        for det in tracks:
            tid = det.track_id
            self._last_seen_ts[tid] = now_ts
            self._seen_frames[tid] = self._seen_frames.get(tid, 0) + 1

            if det.is_vehicle:
                res.active_cars += 1
            elif det.is_person:
                res.active_people += 1

            if tid not in self._counted_tracks and self._seen_frames[tid] >= min_track_frames:
                self._counted_tracks.add(tid)
                if det.is_vehicle:
                    res.new_unique_cars += 1
                    res.new_total_cars += 1
                elif det.is_person:
                    res.new_unique_people += 1
                    res.new_total_people += 1

            status: dict[str, Any] = {
                "in_red_zone": False,
                "dwell_seconds": 0.0,
                "violation": False,
                "zone_name": None,
                "emergency": det.is_emergency,
            }

            if det.is_vehicle and enabled_zones and not (exempt_emergency_vehicles and det.is_emergency):
                nx, ny = det.bottom_center_norm(frame_w, frame_h)
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

        # Clear dwell states for vehicles that left a zone
        for pair in list(self._dwell_states.keys()):
            if pair not in active_pairs:
                del self._dwell_states[pair]

        # Prune old track metadata after 60s of absence
        stale_tids = [tid for tid, ts in self._last_seen_ts.items() if (now_ts - ts) > 60.0]
        for tid in stale_tids:
            self._last_seen_ts.pop(tid, None)
            self._seen_frames.pop(tid, None)
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


class LineCrossingMonitor:
    """Counts vehicles crossing user-defined traffic flow lines and estimates two-line speed."""

    def __init__(self) -> None:
        self._prev_pos: dict[int, tuple[float, float]] = {}
        self._prev_ts: dict[int, float] = {}
        self._last_seen_ts: dict[int, float] = {}
        self._crossed_pairs: set[tuple[int, str]] = set()
        self._first_crossing: dict[int, tuple[str, str, float, tuple[float, float]]] = {}
        self._track_speeds_mph: dict[int, float] = {}
        self._recent_speeds: deque[dict[str, Any]] = deque(maxlen=25)
        self._calibrated: bool = False

    def reset(self) -> None:
        self._prev_pos.clear()
        self._prev_ts.clear()
        self._last_seen_ts.clear()
        self._crossed_pairs.clear()
        self._first_crossing.clear()
        self._track_speeds_mph.clear()
        self._recent_speeds.clear()

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

    def process_tracks(
        self,
        tracks: list[TrackedDetection],
        frame_w: int,
        frame_h: int,
        lines: list[CountingLineModel],
        now_ts: float,
        calib: GroundCalibrationModel | None = None,
    ) -> dict[str, tuple[int, int]]:
        """Return {line_id: (new_a_to_b, new_b_to_a)} and estimate two-line speed when calibrated."""
        enabled_lines = [ln for ln in lines if ln.enabled]
        increments: dict[str, tuple[int, int]] = {}
        self._calibrated = calib is not None
        tf = compute_ground_transforms(calib) if calib is not None else None

        for det in tracks:
            if not det.is_vehicle:
                continue
            tid = det.track_id
            curr = det.bottom_center_norm(frame_w, frame_h)
            prev = self._prev_pos.get(tid)
            prev_ts = self._prev_ts.get(tid, now_ts)
            self._prev_pos[tid] = curr
            self._prev_ts[tid] = now_ts
            self._last_seen_ts[tid] = now_ts

            if prev is None or not enabled_lines:
                continue

            for ln in enabled_lines:
                if (tid, ln.id) in self._crossed_pairs:
                    continue
                p1 = (ln.p1[0], ln.p1[1])
                p2 = (ln.p2[0], ln.p2[1])
                hit = _segment_intersection_point(prev, curr, p1, p2)
                if hit is None:
                    continue
                t_frac, cross_pt = hit
                cross_ts = prev_ts + t_frac * max(0.0, now_ts - prev_ts)

                self._crossed_pairs.add((tid, ln.id))
                side_prev = _ccw(p1[0], p1[1], p2[0], p2[1], prev[0], prev[1])
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
                                self._recent_speeds.append(
                                    {
                                        "track_id": tid,
                                        "vehicle_class": det.class_name,
                                        "speed_mph": mph_rounded,
                                        "from_line": first_name,
                                        "to_line": ln.name,
                                        "distance_ft": round(dist_ft, 1),
                                        "dt_s": round(dt_cross, 2),
                                    }
                                )

        stale_tids = [tid for tid, ts in self._last_seen_ts.items() if (now_ts - ts) > 60.0]
        for tid in stale_tids:
            self._last_seen_ts.pop(tid, None)
            self._prev_pos.pop(tid, None)
            self._prev_ts.pop(tid, None)
            self._first_crossing.pop(tid, None)
            self._track_speeds_mph.pop(tid, None)
            self._crossed_pairs = {pair for pair in self._crossed_pairs if pair[0] != tid}

        return increments


class ObjectDetector:
    """Ultralytics YOLO + built-in ByteTrack wrapper."""

    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        import torch
        from ultralytics import YOLO

        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            import os

            self.device = "cpu"
            torch.set_num_threads(min(4, os.cpu_count() or 4))

        self.model = YOLO(model_name)

    def detect_and_track(
        self, frame: np.ndarray, imgsz: int = 960, conf: float = 0.35
    ) -> list[TrackedDetection]:
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=imgsz,
            conf=conf,
            classes=TARGET_CLASS_IDS,
            device=self.device,
            verbose=False,
        )
        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            return []

        ids = boxes.id.int().cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()
        confs = boxes.conf.float().cpu().tolist()
        xyxys = boxes.xyxy.float().cpu().tolist()

        output: list[TrackedDetection] = []
        for tid, cls_id, score, xyxy in zip(ids, clss, confs, xyxys):
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            cls_name = "person" if cls_id == PERSON_CLASS_ID else VEHICLE_CLASSES.get(cls_id, "vehicle")
            output.append(
                TrackedDetection(
                    track_id=int(tid),
                    class_id=int(cls_id),
                    class_name=cls_name,
                    confidence=float(score),
                    bbox_xyxy=(x1, y1, x2, y2),
                )
            )
        return output


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
