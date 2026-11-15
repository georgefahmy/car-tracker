from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

import requests

from app.config import get_app_version


class HomeAssistantPublisher:
    """Publishes a single Car Tracker device, its entities, and automation events to Home Assistant."""

    DEVICE_NAME = "Car Tracker"
    DEVICE_ID = "car_tracker_addon"
    APP_ID = "io.github.georgefahmy.car_tracker"
    APP_NAME = "Car & Red-Zone Tracker"
    APP_VERSION = get_app_version()
    REGISTRATION_META_KEY = "ha_device_registration"
    REGISTRATION_SCHEMA_VERSION = 1

    LEGACY_ENTITY_IDS = (
        "sensor.car_tracker_vehicles_today",
        "sensor.car_tracker_people_today",
        "sensor.car_tracker_active_vehicles",
        "sensor.car_tracker_last_speed",
        "sensor.car_tracker_avg_speed",
        "sensor.car_tracker_violations_total",
        "sensor.car_tracker_last_plate",
        "binary_sensor.car_tracker_red_zone_occupied",
    )

    def __init__(self, db: Any | None = None) -> None:
        self._db = db
        self._lock = threading.Lock()
        self._publish_lock = threading.Lock()
        self._last_sync_ok: bool | None = None
        self._last_error: str = ""
        self._last_sync_at: str = ""
        self._mode: str = "unconfigured"
        self._entities_count: int = 0
        self._device_registered: bool = False
        self._registration_cache: dict[str, Any] | None = None
        self._pinned_api_base: str = ""
        self._pinned_webhook_base: str = ""
        self._core_direct_bases: list[str] | None = None

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
        reg = self._load_registration()
        with self._lock:
            return {
                "last_sync_ok": self._last_sync_ok,
                "last_error": self._last_error,
                "last_sync_at": self._last_sync_at,
                "mode": self._mode if self._mode != "unconfigured" else (target[2] if target else "unconfigured"),
                "entities_count": self._entities_count,
                "device_name": self.DEVICE_NAME,
                "device_registered": self._device_registered or bool(reg and reg.get("webhook_id")),
                "supervisor_detected": bool(
                    os.environ.get("SUPERVISOR_TOKEN") or os.environ.get("HASSIO_TOKEN")
                ),
            }

    def _load_registration(self) -> dict[str, Any] | None:
        if self._registration_cache and self._registration_cache.get("webhook_id"):
            return dict(self._registration_cache)
        if self._db is not None:
            try:
                raw = self._db.get_meta(self.REGISTRATION_META_KEY)
                if raw:
                    parsed = json.loads(raw)
                    if isinstance(parsed, dict) and parsed.get("webhook_id"):
                        self._registration_cache = dict(parsed)
                        return dict(parsed)
            except Exception:
                return None
        return None

    def _save_registration(self, reg: dict[str, Any] | None) -> None:
        self._registration_cache = dict(reg) if reg else None
        if self._db is not None:
            try:
                self._db.set_meta(self.REGISTRATION_META_KEY, json.dumps(reg or {}))
            except Exception:
                pass

    @staticmethod
    def build_sensor_definitions(payload: dict[str, Any]) -> list[dict[str, Any]]:
        """Build mobile_app sensor & binary_sensor definitions attached to Device 'Car Tracker'."""
        last_speed = payload.get("last_speed_mph")
        avg_speed = payload.get("avg_speed_mph")
        last_plate = str(payload.get("last_plate") or "").strip()
        red_zone_occupied = bool(payload.get("red_zone_occupied", False))

        # Entity names intentionally combine with Device 'Car Tracker' (has_entity_name=True)
        # to produce the exact entity_id slugs: sensor.car_tracker_* and
        # binary_sensor.car_tracker_red_zone_occupied.
        return [
            {
                "type": "sensor",
                "unique_id": "vehicles_today",
                "name": "Vehicles Today",
                "state": int(payload.get("vehicles_today", 0)),
                "unit_of_measurement": "vehicles",
                "state_class": "total_increasing",
                "icon": "mdi:car-multiple",
                "attributes": {
                    "unique_vehicles": int(payload.get("unique_cars", 0)),
                    "active_vehicles": int(payload.get("active_cars", 0)),
                },
            },
            {
                "type": "sensor",
                "unique_id": "people_today",
                "name": "People Today",
                "state": int(payload.get("people_today", 0)),
                "unit_of_measurement": "people",
                "state_class": "total_increasing",
                "icon": "mdi:walk",
                "attributes": {
                    "unique_people": int(payload.get("unique_people", 0)),
                    "active_people": int(payload.get("active_people", 0)),
                },
            },
            {
                "type": "sensor",
                "unique_id": "active_vehicles",
                "name": "Active Vehicles",
                "state": int(payload.get("active_cars", 0)),
                "unit_of_measurement": "vehicles",
                "state_class": "measurement",
                "icon": "mdi:car",
                "attributes": {},
            },
            {
                "type": "sensor",
                "unique_id": "last_speed",
                "name": "Last Speed",
                "state": round(float(last_speed), 1) if last_speed is not None else None,
                "device_class": "speed",
                "unit_of_measurement": "mph",
                "state_class": "measurement",
                "icon": "mdi:speedometer",
                "attributes": {
                    "max_speed_mph": payload.get("max_speed_mph"),
                    "samples_count": int(payload.get("samples_count", 0)),
                },
            },
            {
                "type": "sensor",
                "unique_id": "avg_speed",
                "name": "Avg Speed",
                "state": round(float(avg_speed), 1) if avg_speed is not None else None,
                "device_class": "speed",
                "unit_of_measurement": "mph",
                "state_class": "measurement",
                "icon": "mdi:speedometer-medium",
                "attributes": {},
            },
            {
                "type": "sensor",
                "unique_id": "violations_total",
                "name": "Violations Total",
                "state": int(payload.get("violations_total", 0)),
                "unit_of_measurement": "violations",
                "state_class": "total",
                "icon": "mdi:car-brake-alert",
                "attributes": {
                    "last_violation": payload.get("last_violation"),
                },
            },
            {
                "type": "sensor",
                "unique_id": "last_plate",
                "name": "Last Plate",
                "state": last_plate if last_plate else "none",
                "icon": "mdi:card-text-outline",
                "attributes": {
                    "confidence": payload.get("last_plate_confidence", 0.0),
                    "updated_at": payload.get("last_plate_updated_at", ""),
                },
            },
            {
                "type": "binary_sensor",
                "unique_id": "red_zone_occupied",
                "name": "Red Zone Occupied",
                "state": red_zone_occupied,
                "device_class": "occupancy",
                "icon": "mdi:parking",
                "attributes": {
                    "dwelling_vehicles": int(payload.get("dwelling_vehicles", 0)),
                    "max_dwell_seconds": round(float(payload.get("max_dwell_seconds", 0.0)), 1),
                },
            },
        ]

    @staticmethod
    def build_entities(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
        """Convert live tracker state dict into legacy /api/states payloads (fallback only)."""
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
    def _parse_json_dict(resp: Any) -> dict[str, Any] | None:
        raw_text = getattr(resp, "text", None)
        if isinstance(raw_text, str):
            stripped = raw_text.strip()
            if not stripped:
                return None
            try:
                parsed = json.loads(stripped)
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        json_fn = getattr(resp, "json", None)
        if callable(json_fn):
            try:
                parsed = json_fn()
                return parsed if isinstance(parsed, dict) else None
            except Exception:
                return None
        return None

    @classmethod
    def _post_json_with_fallback(
        cls,
        bases: list[str],
        path: str,
        headers: dict[str, str],
        json_body: dict[str, Any],
    ) -> tuple[bool, int | None, str, dict[str, Any] | None]:
        last_err = ""
        last_code: int | None = None
        for base in bases:
            try:
                resp = requests.post(f"{base}{path}", headers=headers, json=json_body, timeout=4)
                last_code = int(resp.status_code)
                if 200 <= resp.status_code < 300:
                    return (True, last_code, base, cls._parse_json_dict(resp))
                last_err = f"HTTP {resp.status_code}: {str(getattr(resp, 'text', ''))[:160]}"
            except requests.RequestException as exc:
                last_err = str(exc)
        return (False, last_code, last_err or "Home Assistant API request failed", None)

    @classmethod
    def _post_with_fallback(
        cls, bases: list[str], path: str, headers: dict[str, str], json_body: dict[str, Any]
    ) -> tuple[bool, int | None, str]:
        ok, code, info, _data = cls._post_json_with_fallback(bases, path, headers, json_body)
        return (ok, code, info)

    def _ordered_api_bases(self, bases: list[str]) -> list[str]:
        if self._pinned_api_base and self._pinned_api_base in bases:
            return [self._pinned_api_base] + [b for b in bases if b != self._pinned_api_base]
        return list(bases)

    def _discover_core_direct_bases(
        self, api_bases: list[str], auth_headers: dict[str, str]
    ) -> list[str]:
        if self._core_direct_bases is not None:
            return list(self._core_direct_bases)

        discovered: list[str] = []
        for api_base in self._ordered_api_bases(api_bases):
            if "/core/api" not in api_base:
                continue
            sup_root = api_base[: -len("/core/api")]
            try:
                resp = requests.get(f"{sup_root}/core/info", headers=auth_headers, timeout=1.0)
                if int(resp.status_code) == 200:
                    payload = self._parse_json_dict(resp) or {}
                    data = payload.get("data") if isinstance(payload.get("data"), dict) else {}
                    port = int(data.get("port") or 8123)
                    scheme = "https" if bool(data.get("ssl")) else "http"
                    ip_addr = str(data.get("ip_address") or "").strip()
                    discovered.append(f"{scheme}://127.0.0.1:{port}/api")
                    if ip_addr:
                        discovered.append(f"{scheme}://{ip_addr}:{port}/api")
                    break
            except Exception:
                continue

        if discovered:
            self._core_direct_bases = list(dict.fromkeys(discovered))
        return list(self._core_direct_bases or [])

    def _post_webhook(
        self,
        webhook_id: str,
        json_body: dict[str, Any],
        api_bases: list[str],
        mode: str,
        auth_headers: dict[str, str],
    ) -> tuple[str, dict[str, Any] | None, str]:
        """POST a mobile_app webhook payload, handling Supervisor proxy vs direct Core routing.

        Returns ("ok", parsed_dict, working_base), ("invalid_webhook", None, err),
        or ("error", None, err).
        """
        candidates: list[str] = []
        if self._pinned_webhook_base:
            candidates.append(self._pinned_webhook_base)
        for base in self._ordered_api_bases(api_bases):
            if base not in candidates:
                candidates.append(base)
        if mode == "supervisor" and "http://127.0.0.1:8123/api" not in candidates:
            candidates.append("http://127.0.0.1:8123/api")

        last_err = ""
        discovered_extra_added = False
        idx = 0
        while idx < len(candidates):
            wbase = candidates[idx]
            idx += 1
            req_headers = (
                dict(auth_headers)
                if "/core/api" in wbase
                else {"Content-Type": "application/json"}
            )
            try:
                resp = requests.post(
                    f"{wbase}/webhook/{webhook_id}",
                    headers=req_headers,
                    json=json_body,
                    timeout=4,
                )
                status = int(resp.status_code)
                if 200 <= status < 300:
                    parsed = self._parse_json_dict(resp)
                    if parsed:
                        self._pinned_webhook_base = wbase
                        return ("ok", parsed, wbase)
                    return (
                        "invalid_webhook",
                        None,
                        "Home Assistant webhook returned empty response (device may have been deleted)",
                    )
                if status in (401, 403, 404) and "/core/api" in wbase:
                    # Supervisor Core proxy can deny /core/api/webhook/* with 401/403/404;
                    # fall through to direct Home Assistant Core port.
                    last_err = f"HTTP {status}: {str(getattr(resp, 'text', ''))[:160]}"
                elif status in (404, 410):
                    return ("invalid_webhook", None, f"Webhook endpoint returned HTTP {status}")
                else:
                    last_err = f"HTTP {status}: {str(getattr(resp, 'text', ''))[:160]}"
            except requests.RequestException as exc:
                last_err = str(exc)

            if idx >= len(candidates) and mode == "supervisor" and not discovered_extra_added:
                discovered_extra_added = True
                for extra in self._discover_core_direct_bases(api_bases, auth_headers):
                    if extra not in candidates:
                        candidates.append(extra)

        return ("error", None, last_err or "Home Assistant webhook request failed")

    def _cleanup_legacy_states(self, bases: list[str], auth_headers: dict[str, str]) -> None:
        """Delete old loose /api/states/* entities before registering entities under Car Tracker device."""
        active_bases = self._ordered_api_bases(bases)
        for entity_id in self.LEGACY_ENTITY_IDS:
            for base in active_bases:
                try:
                    resp = requests.delete(
                        f"{base}/states/{entity_id}",
                        headers=auth_headers,
                        timeout=3,
                    )
                    if resp.status_code in (200, 204, 404):
                        self._pinned_api_base = base
                        active_bases = [base]
                        break
                except Exception:
                    continue

    def _polish_entities_via_ws(self, active_base: str, token: str) -> None:
        """Best-effort WebSocket polish: hide unused companion device_tracker and set friendly names."""
        try:
            from websockets.sync.client import connect as ws_connect  # type: ignore[import-untyped]
        except Exception:
            return

        if active_base.endswith("/core/api"):
            ws_root = active_base[: -len("/core/api")]
            ws_path = "/core/websocket"
        elif active_base.endswith("/api"):
            ws_root = active_base[: -len("/api")]
            ws_path = "/api/websocket"
        else:
            ws_root = active_base.rstrip("/")
            ws_path = "/api/websocket"

        if ws_root.startswith("https://"):
            ws_url = "wss://" + ws_root[len("https://") :] + ws_path
        elif ws_root.startswith("http://"):
            ws_url = "ws://" + ws_root[len("http://") :] + ws_path
        else:
            return

        try:
            with ws_connect(ws_url, open_timeout=0.5, close_timeout=0.5) as ws:
                hello = json.loads(ws.recv(timeout=1.0))
                if hello.get("type") != "auth_required":
                    return
                ws.send(json.dumps({"type": "auth", "access_token": token}))
                auth_res = json.loads(ws.recv(timeout=1.0))
                if auth_res.get("type") != "auth_ok":
                    return
                updates = [
                    {
                        "id": 1,
                        "type": "config/entity_registry/update",
                        "entity_id": "device_tracker.car_tracker",
                        "disabled_by": "user",
                    },
                    {
                        "id": 2,
                        "type": "config/entity_registry/update",
                        "entity_id": "sensor.car_tracker_avg_speed",
                        "name": "Average Speed",
                    },
                    {
                        "id": 3,
                        "type": "config/entity_registry/update",
                        "entity_id": "sensor.car_tracker_last_plate",
                        "name": "Last License Plate",
                    },
                ]
                for msg in updates:
                    ws.send(json.dumps(msg))
                    ws.recv(timeout=1.0)
        except Exception:
            return

    def _ensure_device_and_sensors(
        self,
        sensors: list[dict[str, Any]],
        bases: list[str],
        token: str,
        mode: str,
        target_key: str,
    ) -> tuple[str, dict[str, Any] | None, str]:
        """Ensure the Car Tracker device and all 8 sensors are registered with mobile_app."""
        auth_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        reg = self._load_registration()
        newly_created_device = False

        if (
            not reg
            or not reg.get("webhook_id")
            or reg.get("target_key") != target_key
            or reg.get("device_name") != self.DEVICE_NAME
        ):
            reg_payload = {
                "device_id": self.DEVICE_ID,
                "app_id": self.APP_ID,
                "app_name": self.APP_NAME,
                "app_version": self.APP_VERSION,
                "device_name": self.DEVICE_NAME,
                "manufacturer": self.APP_NAME,
                "model": "YOLO Vehicle & Curb Monitor",
                "os_name": "Linux",
                "os_version": "Home Assistant Add-on",
                "supports_encryption": False,
            }
            ok, code, info, reg_resp = self._post_json_with_fallback(
                self._ordered_api_bases(bases),
                "/mobile_app/registrations",
                auth_headers,
                reg_payload,
            )
            if not ok:
                if code == 404:
                    return ("unsupported", None, info)
                return ("error", None, info)

            webhook_id = str((reg_resp or {}).get("webhook_id") or "").strip()
            if not webhook_id:
                return ("error", None, "Home Assistant registration response missing webhook_id")

            self._pinned_api_base = info
            self._pinned_webhook_base = ""
            # Remove any leftover loose /api/states/car_tracker_* states so entity IDs stay clean
            self._cleanup_legacy_states([info], auth_headers)

            reg = {
                "schema_version": self.REGISTRATION_SCHEMA_VERSION,
                "device_name": self.DEVICE_NAME,
                "target_key": target_key,
                "webhook_id": webhook_id,
                "registered_ids": [],
            }
            self._save_registration(reg)
            newly_created_device = True

        registered_set = (
            set(reg.get("registered_ids") or [])
            if reg.get("schema_version") == self.REGISTRATION_SCHEMA_VERSION
            else set()
        )
        missing = [s for s in sensors if s["unique_id"] not in registered_set]
        if missing:
            webhook_id = str(reg["webhook_id"])
            for sensor in missing:
                reg_data = {
                    k: v
                    for k, v in sensor.items()
                    if v is not None or k == "state"
                }
                status, _resp, err = self._post_webhook(
                    webhook_id,
                    {"type": "register_sensor", "data": reg_data},
                    bases,
                    mode,
                    auth_headers,
                )
                if status == "invalid_webhook":
                    self._save_registration(None)
                    self._pinned_webhook_base = ""
                    return ("re_register", None, err)
                if status != "ok":
                    return ("error", None, err)
                registered_set.add(sensor["unique_id"])

            reg["schema_version"] = self.REGISTRATION_SCHEMA_VERSION
            reg["registered_ids"] = list(registered_set)
            self._save_registration(reg)

            if newly_created_device:
                ws_base = self._pinned_api_base or bases[0]
                self._polish_entities_via_ws(ws_base, token)

        return ("ok", reg, "")

    def _publish_legacy_states(
        self,
        payload: dict[str, Any],
        bases: list[str],
        token: str,
        mode: str,
    ) -> dict[str, Any]:
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        entities = self.build_entities(payload)
        active_bases = self._ordered_api_bases(bases)
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
                    self._device_registered = False
                return {
                    "ok": False,
                    "error": info,
                    "mode": mode,
                    "entities": pushed,
                    "device_name": self.DEVICE_NAME,
                    "device_registered": False,
                }
            if idx == 0 and info in active_bases:
                self._pinned_api_base = info
                active_bases = [info]
            pushed += 1

        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._last_sync_ok = True
            self._last_error = ""
            self._last_sync_at = now_iso
            self._mode = mode
            self._entities_count = pushed
            self._device_registered = False
        return {
            "ok": True,
            "mode": mode,
            "entities": pushed,
            "device_name": self.DEVICE_NAME,
            "device_registered": False,
            "synced_at": now_iso,
        }

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
                self._device_registered = False
            return {
                "ok": False,
                "error": msg,
                "mode": "unconfigured",
                "entities": 0,
                "device_name": self.DEVICE_NAME,
                "device_registered": False,
            }

        bases, token, mode = target
        target_key = mode if mode == "supervisor" else bases[0]
        auth_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        sensors = self.build_sensor_definitions(payload)

        with self._publish_lock:
            last_err = ""
            for _attempt in range(2):
                ensure_status, reg, ensure_err = self._ensure_device_and_sensors(
                    sensors, bases, token, mode, target_key
                )
                if ensure_status == "unsupported":
                    return self._publish_legacy_states(payload, bases, token, mode)
                if ensure_status == "re_register":
                    last_err = ensure_err
                    continue
                if ensure_status != "ok" or not reg:
                    with self._lock:
                        self._last_sync_ok = False
                        self._last_error = ensure_err
                        self._mode = mode
                        self._device_registered = False
                    return {
                        "ok": False,
                        "error": ensure_err,
                        "mode": mode,
                        "entities": 0,
                        "device_name": self.DEVICE_NAME,
                        "device_registered": False,
                    }

                update_items = [
                    {
                        "type": s["type"],
                        "unique_id": s["unique_id"],
                        "state": s["state"],
                        "icon": s["icon"],
                        "attributes": s["attributes"],
                    }
                    for s in sensors
                ]
                wh_status, wh_resp, wh_err = self._post_webhook(
                    str(reg["webhook_id"]),
                    {"type": "update_sensor_states", "data": update_items},
                    bases,
                    mode,
                    auth_headers,
                )
                if wh_status == "invalid_webhook":
                    self._save_registration(None)
                    self._pinned_webhook_base = ""
                    last_err = wh_err
                    continue
                if wh_status != "ok":
                    with self._lock:
                        self._last_sync_ok = False
                        self._last_error = wh_err
                        self._mode = mode
                    return {
                        "ok": False,
                        "error": wh_err,
                        "mode": mode,
                        "entities": 0,
                        "device_name": self.DEVICE_NAME,
                        "device_registered": True,
                    }

                unregistered_uids: list[str] = []
                if isinstance(wh_resp, dict):
                    for uid, entry in wh_resp.items():
                        if isinstance(entry, dict) and entry.get("success") is False:
                            err_obj = entry.get("error") if isinstance(entry.get("error"), dict) else {}
                            code = str(err_obj.get("code") or "")
                            if "not_registered" in code:
                                unregistered_uids.append(str(uid))

                if unregistered_uids:
                    current_ids = set(reg.get("registered_ids") or [])
                    reg["registered_ids"] = [u for u in current_ids if u not in unregistered_uids]
                    self._save_registration(reg)
                    last_err = f"Re-registering missing sensors: {', '.join(unregistered_uids)}"
                    continue

                now_iso = datetime.now(timezone.utc).isoformat()
                with self._lock:
                    self._last_sync_ok = True
                    self._last_error = ""
                    self._last_sync_at = now_iso
                    self._mode = mode
                    self._entities_count = len(sensors)
                    self._device_registered = True
                return {
                    "ok": True,
                    "mode": mode,
                    "entities": len(sensors),
                    "device_name": self.DEVICE_NAME,
                    "device_registered": True,
                    "synced_at": now_iso,
                }

            err_final = last_err or "Home Assistant device sync failed"
            with self._lock:
                self._last_sync_ok = False
                self._last_error = err_final
                self._mode = mode
                self._device_registered = False
            return {
                "ok": False,
                "error": err_final,
                "mode": mode,
                "entities": 0,
                "device_name": self.DEVICE_NAME,
                "device_registered": False,
            }

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
        ok, _code, _info = self._post_with_fallback(
            self._ordered_api_bases(bases), f"/events/{event_type}", headers, event_data
        )
        return ok
