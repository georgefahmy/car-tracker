from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

from app.database import CountingLineModel, ZoneModel

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


class LineCrossingMonitor:
    """Counts vehicles crossing user-defined traffic flow lines in either direction."""

    def __init__(self) -> None:
        self._prev_pos: dict[int, tuple[float, float]] = {}
        self._last_seen_ts: dict[int, float] = {}
        self._crossed_pairs: set[tuple[int, str]] = set()

    def reset(self) -> None:
        self._prev_pos.clear()
        self._last_seen_ts.clear()
        self._crossed_pairs.clear()

    def process_tracks(
        self,
        tracks: list[TrackedDetection],
        frame_w: int,
        frame_h: int,
        lines: list[CountingLineModel],
        now_ts: float,
    ) -> dict[str, tuple[int, int]]:
        """Return {line_id: (new_a_to_b, new_b_to_a)} for vehicles crossing enabled lines."""
        enabled_lines = [ln for ln in lines if ln.enabled]
        increments: dict[str, tuple[int, int]] = {}

        for det in tracks:
            if not det.is_vehicle:
                continue
            tid = det.track_id
            curr = det.bottom_center_norm(frame_w, frame_h)
            prev = self._prev_pos.get(tid)
            self._prev_pos[tid] = curr
            self._last_seen_ts[tid] = now_ts

            if prev is None or not enabled_lines:
                continue

            for ln in enabled_lines:
                if (tid, ln.id) in self._crossed_pairs:
                    continue
                p1 = (ln.p1[0], ln.p1[1])
                p2 = (ln.p2[0], ln.p2[1])
                if _segments_intersect(prev, curr, p1, p2):
                    self._crossed_pairs.add((tid, ln.id))
                    side_prev = _ccw(p1[0], p1[1], p2[0], p2[1], prev[0], prev[1])
                    a_inc, b_inc = increments.get(ln.id, (0, 0))
                    if side_prev < 0:
                        increments[ln.id] = (a_inc + 1, b_inc)
                    else:
                        increments[ln.id] = (a_inc, b_inc + 1)

        stale_tids = [tid for tid, ts in self._last_seen_ts.items() if (now_ts - ts) > 60.0]
        for tid in stale_tids:
            self._last_seen_ts.pop(tid, None)
            self._prev_pos.pop(tid, None)
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
