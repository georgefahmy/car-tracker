from __future__ import annotations

import os
import re
import threading
from typing import Any

import cv2
import numpy as np

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")

_PLATE_BLOCKLIST = {
    "REOLINK",
    "CAMERA",
    "CHANNEL",
    "SPEED",
    "LIMIT",
    "EXEMPT",
    "POLICE",
    "ILLEGAL",
    "PARK",
    "MPH",
    "STOP",
    "TRUCK",
    "VEHICLE",
    "TOYOTA",
    "HONDA",
    "TESLA",
    "SUBARU",
    "NISSAN",
    "HYUNDAI",
    "FORD",
    "CHEVY",
    "BMW",
    "AUDI",
    "LEXUS",
    "2160P",
    "1080P",
    "720P",
}


def normalize_plate_candidate(raw_text: str, confidence: float = 1.0) -> str | None:
    """Normalize an OCR string into a plausible license plate or return None."""
    if not raw_text:
        return None
    upper = raw_text.upper().strip()

    # Evaluate both the joined text (handles "7ABC 123" / "7ABC-123") and individual tokens
    # (handles state banners like "CA 7ABC123").
    parts = [p for p in re.split(r"\s+", upper) if p]
    candidates: list[str] = []
    joined = _NON_ALNUM_RE.sub("", upper)
    if joined:
        candidates.append(joined)
    for part in reversed(parts):
        cleaned_part = _NON_ALNUM_RE.sub("", part)
        if cleaned_part and cleaned_part not in candidates:
            candidates.append(cleaned_part)

    for cleaned in candidates:
        if cleaned in _PLATE_BLOCKLIST:
            continue
        if cleaned.isdigit() and len(cleaned) == 4 and 1990 <= int(cleaned) <= 2099:
            continue
        has_digit = any(c.isdigit() for c in cleaned)
        has_alpha = any(c.isalpha() for c in cleaned)
        if 4 <= len(cleaned) <= 8 and has_digit and has_alpha:
            return cleaned
        if 5 <= len(cleaned) <= 7 and cleaned.isdigit() and confidence >= 0.85:
            return cleaned
    return None


class LicensePlateReader:
    """Lazy-loaded local ONNX OCR reader run strictly on-demand for violation/speeding crops."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ocr: Any = None
        self._init_failed = False

    def _ensure_ocr(self) -> Any:
        if self._ocr is not None:
            return self._ocr
        if self._init_failed:
            return None
        with self._lock:
            if self._ocr is not None:
                return self._ocr
            try:
                from rapidocr_onnxruntime import RapidOCR

                ocr_threads = min(2, max(1, os.cpu_count() or 2))
                self._ocr = RapidOCR(
                    intra_op_num_threads=ocr_threads,
                    inter_op_num_threads=1,
                    use_cls=False,
                )
            except Exception as exc:
                self._init_failed = True
                print(f"[alpr] RapidOCR initialization unavailable: {exc}", flush=True)
                return None
        return self._ocr

    @staticmethod
    def crop_vehicle_region(
        frame: np.ndarray, bbox_xyxy: tuple[float, float, float, float] | None
    ) -> np.ndarray:
        """Crop the vehicle region with light padding and upscale small crops for legible plates."""
        if frame is None or frame.size == 0:
            return frame
        h, w = frame.shape[:2]
        if bbox_xyxy is not None:
            x1, y1, x2, y2 = [float(v) for v in bbox_xyxy]
            bw = max(x2 - x1, 0.0)
            bh = max(y2 - y1, 0.0)
            if bw >= 12.0 and bh >= 12.0:
                cx1 = max(0, int(round(x1 - 0.08 * bw)))
                # Trim the top 15% (roof rack / cabin roof / windshield decals) when tall enough
                cy1 = max(0, int(round(y1 + (0.15 * bh if bh >= 32.0 else 0.0))))
                cx2 = min(w, int(round(x2 + 0.08 * bw)))
                cy2 = min(h, int(round(y2 + 0.10 * bh)))
                if cx2 > cx1 and cy2 > cy1:
                    crop = frame[cy1:cy2, cx1:cx2]
                else:
                    crop = frame
            else:
                crop = frame
        else:
            crop = frame

        ch, cw = crop.shape[:2]
        if 0 < cw < 360:
            scale = 360.0 / float(cw)
            crop = cv2.resize(
                crop,
                (360, max(1, int(round(ch * scale)))),
                interpolation=cv2.INTER_CUBIC,
            )
        elif cw > 960:
            scale = 960.0 / float(cw)
            crop = cv2.resize(
                crop,
                (960, max(1, int(round(ch * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        return crop

    @staticmethod
    def _enhance_contrast(img: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        merged = cv2.merge((clahe.apply(l_chan), a_chan, b_chan))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    @staticmethod
    def _select_best_candidate(
        ocr_result: list[Any] | None, crop_h: int
    ) -> tuple[str, float]:
        if not ocr_result:
            return ("", 0.0)

        best_plate = ""
        best_conf = 0.0
        best_score = -1.0

        for item in ocr_result:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue
            box, raw_text, raw_conf = item[0], str(item[1]), float(item[2] or 0.0)
            plate = normalize_plate_candidate(raw_text, confidence=raw_conf)
            if not plate:
                continue

            has_both = any(c.isdigit() for c in plate) and any(c.isalpha() for c in plate)
            len_bonus = 0.18 if 5 <= len(plate) <= 7 else 0.08
            mix_bonus = 0.25 if has_both else 0.0

            y_bonus = 0.0
            if isinstance(box, (list, tuple)) and box:
                try:
                    ys = [float(pt[1]) for pt in box if isinstance(pt, (list, tuple)) and len(pt) >= 2]
                    if ys and crop_h > 0:
                        rel_y = max(0.0, min(1.0, (sum(ys) / len(ys)) / float(crop_h)))
                        y_bonus = 0.12 * rel_y
                except Exception:
                    pass

            score = raw_conf + mix_bonus + len_bonus + y_bonus
            if score > best_score:
                best_score = score
                best_plate = plate
                best_conf = raw_conf

        return (best_plate, round(best_conf, 3)) if best_plate else ("", 0.0)

    def read_plate(
        self,
        frame: np.ndarray,
        bbox_xyxy: tuple[float, float, float, float] | None = None,
    ) -> tuple[str, float]:
        """Run on-demand ALPR on a clean high-res frame + vehicle box. Never raises."""
        try:
            if frame is None or frame.size == 0:
                return ("", 0.0)
            ocr = self._ensure_ocr()
            if ocr is None:
                return ("", 0.0)

            crop = self.crop_vehicle_region(frame, bbox_xyxy)
            res, _ = ocr(crop)
            plate, conf = self._select_best_candidate(res, crop.shape[0])
            if plate:
                return (plate, conf)

            # Single contrast-enhanced fallback on lower 80% (where front/rear bumpers sit)
            ch = crop.shape[0]
            lower = crop[int(ch * 0.20) :, :] if ch >= 40 else crop
            enhanced = self._enhance_contrast(lower)
            res2, _ = ocr(enhanced)
            return self._select_best_candidate(res2, enhanced.shape[0])
        except Exception as exc:
            print(f"[alpr] OCR skipped due to error: {exc}", flush=True)
            return ("", 0.0)
