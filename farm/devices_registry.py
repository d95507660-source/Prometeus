"""devices_registry.py — реестр устройств и фильтр «по точке» (маркер ниши в username)."""
from __future__ import annotations

import ipaddress
import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
HOST_RE = re.compile(r"^[A-Za-z0-9.\-]{1,253}$")


class DevicesRegistry:
    def __init__(self, config: dict, devices_file: str | Path = "devices.json",
                 niches_file: str | Path = "niches.json"):
        self.cfg = config
        self.devices: list[dict] = json.loads(
            Path(devices_file).read_text(encoding="utf-8")).get("devices", [])
        niches_path = Path(niches_file)
        self.niches: dict = (json.loads(niches_path.read_text(encoding="utf-8")).get("niches", {})
                             if niches_path.exists() else {})
        reg = config.get("devices_registry", {})
        self.require_marker = bool(reg.get("require_username_with_niche_marker", True))
        self.markers: dict = reg.get("niche_markers", {})

    @classmethod
    def from_config(cls, cfg: Any) -> "DevicesRegistry":
        """Создать реестр из AppConfig (пути берутся от корня проекта)."""
        return cls(cfg.raw, cfg.path("devices"), cfg.path("niches"))

    # ------------------------------------------------------------------
    def _marker_for(self, niche: str) -> str:
        return self.markers.get(niche) or self.niches.get(niche, {}).get("marker", "")

    def username_problem(self, username: str, niche: str) -> str | None:
        marker = self._marker_for(niche)
        if not self.require_marker:
            return None
        if not marker:
            return f"для ниши '{niche}' не задан маркер"
        if marker.lower() not in username.lower():
            return f"username '{username}' не содержит маркер '{marker}'"
        return None

    @staticmethod
    def _adb_problem(device: dict) -> str | None:
        adb = device.get("adb") or {}
        host = str(adb.get("host") or "").strip()
        port = adb.get("port")
        if not host:
            return "adb.host пустой"
        if not HOST_RE.match(host):
            return f"adb.host некорректный: {host!r}"
        try:
            ipaddress.ip_address(host)
        except ValueError:
            pass  # допускаем имя хоста
        try:
            port_i = int(port)
        except (TypeError, ValueError):
            return f"adb.port некорректный: {port!r}"
        if not 1 <= port_i <= 65535:
            return f"adb.port вне диапазона: {port_i}"
        return None

    def get_active_devices(self, niches: list[str] | None = None,
                           device_ids: list[str] | None = None) -> list[dict]:
        active_niches = set(niches or self.cfg.get("mimic", {}).get("active_niches", []))
        wanted = set(device_ids or [])
        out: list[dict] = []
        seen_ids: set[str] = set()
        seen_adb: dict[str, str] = {}
        for d in self.devices:
            did = str(d.get("device_id") or "")
            if wanted and did not in wanted:
                continue
            if not d.get("enabled", True):
                continue
            niche = d.get("niche")
            if active_niches and niche not in active_niches:
                continue
            problems: list[str] = []
            if not DEVICE_ID_RE.match(did):
                problems.append(f"device_id некорректный: {did!r} (только латиница, цифры, _ и -)")
            if did in seen_ids:
                problems.append("device_id повторяется")
            p = self._adb_problem(d)
            if p:
                problems.append(p)
            else:
                serial = f"{d['adb']['host']}:{int(d['adb']['port'])}"
                if serial in seen_adb:
                    problems.append(f"adb {serial} уже занят устройством {seen_adb[serial]}")
            enabled_accounts = 0
            for platform, acc in (d.get("accounts") or {}).items():
                if not acc.get("enabled", True):
                    continue
                enabled_accounts += 1
                if not acc.get("username"):
                    problems.append(f"{platform}: пустой username")
                    continue
                up = self.username_problem(acc["username"], niche)
                if up:
                    problems.append(f"{platform}: {up}")
            if enabled_accounts == 0:
                problems.append("нет ни одного включённого аккаунта")
            if problems:
                log.warning("устройство %s отброшено: %s", did or "?", "; ".join(problems))
                continue
            seen_ids.add(did)
            seen_adb[f"{d['adb']['host']}:{int(d['adb']['port'])}"] = did
            out.append(d)
        if wanted:
            missing = wanted - seen_ids
            if missing:
                log.warning("запрошенные устройства не найдены или отброшены: %s", sorted(missing))
        return out


if __name__ == "__main__":
    from config_loader import cli_config
    cfg, _ = cli_config("devices_registry")
    reg = DevicesRegistry.from_config(cfg)
    active = reg.get_active_devices()
    for d in active:
        accs = [a.get("username") for a in (d.get("accounts") or {}).values() if a.get("enabled", True)]
        print(d["device_id"], d["niche"], accs)
    print(f"активных устройств: {len(active)} из {len(reg.devices)}")
