"""diversity.py — общие детерминированные помощники разнообразия контента.

Используется forge.py, mimic.py, ghost.py (оркестрация на ноутбуке).
В контейнер model_server этот модуль НЕ нужен: там своя минимальная логика сида лица.

Когорта — все активные устройства одной ниши, отсортированные по device_id.
Ранг устройства в когорте даёт гарантированно разные значения (хуки, стиль), а не
«почти разные», как при hash(device_id) % len. Если в нишу добавить устройство,
ранги сдвинутся — уже сохранённые планы mimic при этом не перегенерируются (см. mimic.py).
"""
from __future__ import annotations

import hashlib
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Iterable

DEVICE_ID_RE = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
DAY_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def validate_device_id(device_id: Any) -> str:
    """device_id попадает в пути и URI GCS — пропускаем только [A-Za-z0-9_-]{1,64}."""
    if not isinstance(device_id, str) or not DEVICE_ID_RE.fullmatch(device_id):
        raise ValueError(f"некорректный device_id: {device_id!r}")
    return device_id


def validate_day(day: Any) -> str:
    if not isinstance(day, str) or not DAY_RE.fullmatch(day):
        raise ValueError(f"некорректная дата (ожидалось YYYY-MM-DD): {day!r}")
    return day


def stable_int(*parts: Any) -> int:
    """32-битное число, одинаковое между запусками (hash() зависит от PYTHONHASHSEED)."""
    return int(hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).hexdigest()[:8], 16)


def niche_cohorts(devices: Iterable[dict]) -> dict[str, list[str]]:
    """{niche: [device_id, ...]} — отсортировано, без дублей."""
    out: dict[str, set[str]] = {}
    for d in devices:
        did, niche = d.get("device_id"), d.get("niche")
        if did and niche:
            out.setdefault(str(niche), set()).add(str(did))
    return {k: sorted(v) for k, v in out.items()}


def cohort_rank(cohort: Iterable[str], device_id: str) -> tuple[int, int]:
    """(ранг, размер когорты). Устройство, которого нет в когорте, добавляется в неё."""
    members = sorted(set(cohort) | {device_id})
    return members.index(device_id), len(members)


def permutation(n: int, *seed_parts: Any) -> list[int]:
    """Детерминированная перестановка range(n)."""
    idx = list(range(max(0, n)))
    random.Random(stable_int(*seed_parts)).shuffle(idx)
    return idx


def atomic_write_text(path: Path, text: str) -> None:
    """tmp в той же папке + fsync + os.replace: читатель видит либо старый, либо новый файл."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except FileNotFoundError:
            pass
        raise
