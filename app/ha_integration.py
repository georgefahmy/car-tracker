from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from typing import Any

import requests


class HomeAssistantPublisher:
    """Publishes Car Tracker entities and automation events to Home Assistant Core REST API."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_sync_ok: bool | None = None
        self._last_error: str = ""
        self._last_sync_at: str = ""
        self._mode: str = "unconfigured"
        self._entities_count: int = 0

    @staticmethod
    def resolve_target(
        ha_url: str = "", ha_token: str = ""
    ) -> tuple[list[str], str, str] | None:
        """Resolve candidate HA Core API base URLs, Bearer token, and mode."""
        clean_url = (ha_url or "").strip().rstrip("/")
        clean_token = (ha_token or "").strip()
        if clean_url and clean_token:
            base = clean_url if clean_url.endswith("/api") else f"{clean_url}/api"
            return ([base], clean_token, "custom")

        sup_token = (os.environ.get("SUPERVISOR_TOKEN") or os.environ.get("HASSIO_TOKEN") or "").strip()
        if sup_token:
            # host_network: true containers may lack Docker internal DNS for `supervisor`,
            # so include Supervisor's default link-local/gateway IP as a seamless fallback.
            return (["http://supervisor/core/api", "http://172.30.32.2/core/api"], sup_token, "supervisor")
        return None

    def get_status(self) -> dict[str, Any]:
        target = self.resolve_target()
        with self._lock:
            return {
                "last_sync_ok": self._last_sync_ok,
                "last_error": self._last_error,
                "last_sync_at": self._last_sync_at,
                "mode": self._mode if self._mode != "unconfigured" else (target[2] if target else "unconfigured"),
                "entities_count": self._entities_count,
                "supervisor_detected": bool(
                    os.environ.get("SUPERVISOR_TOKEN") or os.environ.get("HASSIO_TOKEN")
                ),
            }

    @staticmethod
    def build_entities(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """Convert live tracker state dict into Home Assistant entity state payloads."""
        last_speed = payload.get("last_speed_mph")
        avg_speed = payload.get("avg_speed_mph")
        last_plate = str(payload.get("last_plate") or "").strip()
        red_zone_occupied = bool(payload.get("red_zone_occupied", False))

        return [
            (
                "sensor.car_tracker_vehicles_today",
                {
                    "state": int(payload.get("vehicles_today", 0)),
                    "attributes": {
                        "friendly_name": "Car Tracker Vehicles Today",
                        "unit_of_measurement": "vehicles",
                        "state_class": "total_increasing",
                        "icon": "mdi:car-multiple",
                        "unique_vehicles": int(payload.get("unique_cars", 0)),
                        "active_vehicles": int(payload.get("active_cars", 0)),
                    },
                },
            ),
            (
                "sensor.car_tracker_people_today",
                {
                    "state": int(payload.get("people_today", 0)),
                    "attributes": {
                        "friendly_name": "Car Tracker People Today",
                        "unit_of_measurement": "people",
                        "state_class": "total_increasing",
                        "icon": "mdi:walk",
                        "unique_people": int(payload.get("unique_people", 0)),
                        "active_people": int(payload.get("active_people", 0)),
                    },
                },
            ),
            (
                "sensor.car_tracker_active_vehicles",
                {
                    "state": int(payload.get("active_cars", 0)),
                    "attributes": {
                        "friendly_name": "Car Tracker Active Vehicles",
                        "unit_of_measurement": "vehicles",
                        "state_class": "measurement",
                        "icon": "mdi:car",
                    },
                },
            ),
            (
                "sensor.car_tracker_last_speed",
                {
                    "state": round(float(last_speed), 1) if last_speed is not None else "unknown",
                    "attributes": {
                        "friendly_name": "Car Tracker Last Speed",
                        "unit_of_measurement": "mph",
                        "device_class": "speed",
                        "state_class": "measurement",
                        "icon": "mdi:speedometer",
                        "max_speed_mph": payload.get("max_speed_mph"),
                        "samples_count": int(payload.get("samples_count", 0)),
                    },
                },
            ),
            (
                "sensor.car_tracker_avg_speed",
                {
                    "state": round(float(avg_speed), 1) if avg_speed is not None else "unknown",
                    "attributes": {
                        "friendly_name": "Car Tracker Average Speed",
                        "unit_of_measurement": "mph",
                        "device_class": "speed",
                        "state_class": "measurement",
                        "icon": "mdi:speedometer-medium",
                    },
                },
            ),
            (
                "sensor.car_tracker_violations_total",
                {
                    "state": int(payload.get("violations_total", 0)),
                    "attributes": {
                        "friendly_name": "Car Tracker Violations Total",
                        "unit_of_measurement": "violations",
                        "state_class": "total",
                        "icon": "mdi:car-brake-alert",
                        "last_violation": payload.get("last_violation"),
                    },
                },
            ),
            (
                "sensor.car_tracker_last_plate",
                {
                    "state": last_plate if last_plate else "none",
                    "attributes": {
                        "friendly_name": "Car Tracker Last License Plate",
                        "icon": "mdi:card-text-outline",
                        "confidence": payload.get("last_plate_confidence", 0.0),
                        "updated_at": payload.get("last_plate_updated_at", ""),
                    },
                },
            ),
            (
                "binary_sensor.car_tracker_red_zone_occupied",
                {
                    "state": "on" if red_zone_occupied else "off",
                    "attributes": {
                        "friendly_name": "Car Tracker Red Zone Occupied",
                        "device_class": "occupancy",
                        "icon": "mdi:parking",
                        "dwelling_vehicles": int(payload.get("dwelling_vehicles", 0)),
                        "max_dwell_seconds": round(float(payload.get("max_dwell_seconds", 0.0)), 1),
                    },
                },
            ),
        ]

    @staticmethod
    def _post_with_fallback(
        bases: list[str], path: str, headers: dict[str, str], json_body: dict[str, Any]
    ) -> tuple[bool, int | None, str]:
        last_err = ""
        for base in bases:
            try:
                resp = requests.post(f"{base}{path}", headers=headers, json=json_body, timeout=4)
                if 200 <= resp.status_code < 300:
                    return (True, resp.status_code, base)
                last_err = f"HTTP {resp.status_code}: {resp.text[:160]}"
            except requests.RequestException as exc:
                last_err = str(exc)
        return (False, None, last_err or "Home Assistant API request failed")

    def publish_states(
        self, payload: dict[str, Any], ha_url: str = "", ha_token: str = ""
    ) -> dict[str, Any]:
        target = self.resolve_target(ha_url=ha_url, ha_token=ha_token)
        if target is None:
            msg = "Not running under HA Supervisor and no custom HA URL/Token configured."
            with self._lock:
                self._last_sync_ok = False
                self._last_error = msg
                self._mode = "unconfigured"
            return {"ok": False, "error": msg, "mode": "unconfigured", "entities": 0}

        bases, token, mode = target
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        entities = self.build_entities(payload)

        active_bases = list(bases)
        pushed = 0
        for idx, (entity_id, body) in enumerate(entities):
            ok, _code, info = self._post_with_fallback(
                active_bases, f"/states/{entity_id}", headers, body
            )
            if not ok:
                with self._lock:
                    self._last_sync_ok = False
                    self._last_error = info
                    self._mode = mode
                return {"ok": False, "error": info, "mode": mode, "entities": pushed}
            if idx == 0 and info in active_bases:
                # Pin the working base URL first for the remaining entity pushes in this batch
                active_bases = [info]
            pushed += 1

        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._last_sync_ok = True
            self._last_error = ""
            self._last_sync_at = now_iso
            self._mode = mode
            self._entities_count = pushed
        return {"ok": True, "mode": mode, "entities": pushed, "synced_at": now_iso}

    def fire_event(
        self,
        event_type: str,
        event_data: dict[str, Any],
        ha_url: str = "",
        ha_token: str = "",
    ) -> bool:
        target = self.resolve_target(ha_url=ha_url, ha_token=ha_token)
        if target is None:
            return False
        bases, token, _mode = target
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        ok, _code, _info = self._post_with_fallback(bases, f"/events/{event_type}", headers, event_data)
        return ok
