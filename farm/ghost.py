"""ghost.py — подготовка source_N.mp4 под TikTok / Instagram / YouTube (9:16, 1080×1920)
в фирменном визуальном стиле канала.

Вход:  data/videos/<device_id>/<YYYY-MM-DD>/source_<N>.mp4
Выход: data/videos/<device_id>/<YYYY-MM-DD>/source_<N>/<platform>.mp4
       (+ _styled.mp4 и _style.json — общий рендер и его параметры)

Стиль канала = пресет цветокоррекции (6) × сила (3) × кадрирование (2) = 36 заметных глазу
вариантов. Внутри ниши стиль назначается по рангу аккаунта в когорте, поэтому до 36
аккаунтов в нише получают разные сочетания; соседние по рангу отличаются пресетом.
Без когорты — по hash(device_id) (совпадения возможны).

Что убрано: случайный сдвиг скорости 0.97–1.03, яркости ±3 % и зума на каждую площадку.
Зритель такие сдвиги не замечает — они только меняют «отпечаток» файла, т. е. работают на
сокрытие дубликатов, а не на разнообразие; atempo к тому же портил озвучку. Теперь ролик
одного аккаунта на всех площадках одинаковый (один рендер, копии по площадкам).
"""
from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from diversity import atomic_write_text, cohort_rank, niche_cohorts, stable_int, validate_device_id

log = logging.getLogger(__name__)

OUT_W, OUT_H = 1080, 1920
PLATFORM_RE = re.compile(r"^[a-z][a-z0-9_]{1,19}$")
DAY_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Отклонения от нейтрали; сила (strength) масштабирует их.
# eq: contrast/saturation/gamma — множители (1.0 = нейтраль), brightness — сдвиг.
# colorbalance: rs/gs/bs — тени, rm/gm/bm — средние тона, rh/gh/bh — света, диапазон −1…1.
STYLE_PRESETS: dict[str, dict] = {
    "warm_film":   {"contrast": 1.06, "saturation": 0.92, "gamma": 1.00, "brightness": 0.01,
                    "cb": {"rs": 0.06, "gs": 0.02, "bs": -0.06, "rm": 0.05, "bm": -0.05,
                           "rh": 0.03, "bh": -0.04}, "vignette": True},
    "cool_crisp":  {"contrast": 1.10, "saturation": 1.04, "gamma": 0.98, "brightness": 0.00,
                    "cb": {"rs": -0.04, "bs": 0.06, "rm": -0.03, "bm": 0.04, "bh": 0.02},
                    "vignette": False},
    "vivid_pop":   {"contrast": 1.08, "saturation": 1.28, "gamma": 1.00, "brightness": 0.02,
                    "cb": {}, "vignette": False},
    "soft_pastel": {"contrast": 0.90, "saturation": 0.82, "gamma": 1.06, "brightness": 0.03,
                    "cb": {"rh": 0.03, "bh": 0.03, "gm": 0.02}, "vignette": False},
    "moody_matte": {"contrast": 0.88, "saturation": 0.85, "gamma": 0.96, "brightness": -0.03,
                    "cb": {"bs": 0.06, "gs": 0.02, "rh": 0.03}, "vignette": True},
    "golden_hour": {"contrast": 1.04, "saturation": 1.12, "gamma": 1.02, "brightness": 0.02,
                    "cb": {"rm": 0.07, "gm": 0.03, "bm": -0.07, "rh": 0.05}, "vignette": False},
}
PRESET_NAMES = list(STYLE_PRESETS)  # порядок фиксирован определением словаря
STRENGTHS: tuple[tuple[str, float], ...] = (("light", 0.5), ("medium", 1.0), ("strong", 1.5))
FRAMINGS: tuple[tuple[str, float], ...] = (("wide", 1.00), ("tight", 1.08))
STYLE_COMBOS = len(PRESET_NAMES) * len(STRENGTHS) * len(FRAMINGS)


@dataclass(frozen=True)
class Style:
    index: int
    preset: str
    strength: str
    framing: str

    def key(self) -> str:
        return f"{self.preset}/{self.strength}/{self.framing}"


def style_for_index(index: int) -> Style:
    i = int(index) % STYLE_COMBOS
    p = len(PRESET_NAMES)
    preset = PRESET_NAMES[i % p]
    strength = STRENGTHS[(i // p) % len(STRENGTHS)][0]
    framing = FRAMINGS[(i // (p * len(STRENGTHS))) % len(FRAMINGS)][0]
    return Style(i, preset, strength, framing)


def style_for_device(device_id: str, cohort: list[str] | None = None,
                     style_index: int | None = None) -> Style:
    """style_index (из devices.json) > ранг в когорте ниши > hash(device_id)."""
    validate_device_id(device_id)
    if style_index is not None:
        return style_for_index(int(style_index))
    if cohort:
        rank, n = cohort_rank(cohort, device_id)
        if n > STYLE_COMBOS:
            log.warning("ghost: в нише %d аккаунтов > %d стилей — стили будут повторяться",
                        n, STYLE_COMBOS)
        return style_for_index(rank)
    return style_for_index(stable_int(device_id, "style"))


def build_filter(style: Style) -> str:
    """Цепочка -vf для ffmpeg. Только числа из таблиц — никаких внешних строк."""
    k = dict(STRENGTHS)[style.strength]
    zoom = dict(FRAMINGS)[style.framing]
    pr = STYLE_PRESETS[style.preset]

    def mul(v: float) -> float:
        return 1.0 + (v - 1.0) * k

    zw = int(OUT_W * zoom) // 2 * 2
    zh = int(OUT_H * zoom) // 2 * 2
    parts = [f"scale={zw}:{zh}:force_original_aspect_ratio=increase",
             f"crop={OUT_W}:{OUT_H}", "setsar=1",
             (f"eq=contrast={mul(pr['contrast']):.3f}:brightness={pr['brightness'] * k:.3f}"
              f":saturation={mul(pr['saturation']):.3f}:gamma={mul(pr['gamma']):.3f}")]
    cb = {name: max(-1.0, min(1.0, v * k)) for name, v in pr["cb"].items()}
    if cb:
        parts.append("colorbalance=" + ":".join(f"{n}={v:.3f}" for n, v in sorted(cb.items())))
    if pr["vignette"]:
        parts.append(f"vignette=angle={0.25 + 0.35 * k:.3f}")
    parts.append("format=yuv420p")
    return ",".join(parts)


def _run(cmd: list[str], timeout: int = 900) -> None:
    log.debug("ffmpeg: %s", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode("utf-8", "ignore")[-800:])


def _has_audio(src: Path) -> bool:
    p = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a",
                        "-show_entries", "stream=index", "-of", "csv=p=0", str(src)],
                       capture_output=True, text=True, timeout=60)
    return bool(p.stdout.strip())


def _style_meta(out_dir: Path) -> dict:
    f = out_dir / "_style.json"
    try:
        return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    except Exception:
        return {}


def render_styled(src: Path, out_dir: Path, style: Style, force: bool = False) -> Path:
    """Один рендер на исходник: out_dir/_styled.mp4. Перерендер — если сменился стиль
    или исходник новее."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "_styled.mp4"
    meta = _style_meta(out_dir)
    if (not force and dst.exists() and dst.stat().st_size > 0
            and dst.stat().st_mtime >= src.stat().st_mtime and meta.get("style") == style.key()):
        return dst
    tmp = out_dir / "_styled.tmp.mp4"
    cmd = ["ffmpeg", "-hide_banner", "-y", "-i", str(src), "-vf", build_filter(style)]
    if _has_audio(src):
        cmd += ["-c:a", "aac", "-b:a", "128k"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-r", "30",
            "-movflags", "+faststart", str(tmp)]
    try:
        _run(cmd)
        os.replace(tmp, dst)  # не оставляем битый файл, если ffmpeg упал
    finally:
        tmp.unlink(missing_ok=True)
    atomic_write_text(out_dir / "_style.json",
                      json.dumps({"style": style.key(), "index": style.index}, ensure_ascii=False))
    log.info("ghost %s -> %s (стиль %s)", src.name, dst, style.key())
    return dst


def _copy_atomic(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.tmp")
    try:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)
    finally:
        tmp.unlink(missing_ok=True)


def uniquify(src: Path, out_dir: Path, platform: str, seed: int | None = None,
             force: bool = False, style: Style | None = None) -> Path:
    """Имя оставлено для совместимости. Готовит out_dir/<platform>.mp4 в стиле канала.
    style не задан → стиль по seed (или по имени файла)."""
    if not PLATFORM_RE.fullmatch(platform or ""):
        raise ValueError(f"некорректное имя площадки: {platform!r}")
    if style is None:
        style = style_for_index(seed if seed is not None else stable_int(src.name, "style"))
    styled = render_styled(src, out_dir, style, force=force)
    dst = out_dir / f"{platform}.mp4"
    if force or not dst.exists() or dst.stat().st_size == 0 \
            or dst.stat().st_mtime < styled.stat().st_mtime:
        _copy_atomic(styled, dst)
    return dst


def uniquify_device(device_id: str, videos_root: Path, platforms: list[str],
                    source_idx: int | None = None, day: str | None = None,
                    force: bool = False, cohort: list[str] | None = None,
                    style_index: int | None = None) -> dict[str, str]:
    """day=None — все дни. cohort — device_id ниши: даёт гарантированно разные стили."""
    validate_device_id(device_id)
    dev_dir = Path(videos_root) / device_id
    if not dev_dir.exists():
        log.warning("ghost: нет папки %s", dev_dir)
        return {}
    if day is not None and not DAY_DIR_RE.fullmatch(day):
        raise ValueError(f"некорректная дата: {day!r}")
    style = style_for_device(device_id, cohort, style_index)
    day_dirs = [dev_dir / day] if day else sorted(
        p for p in dev_dir.iterdir() if p.is_dir() and DAY_DIR_RE.fullmatch(p.name))
    result: dict[str, str] = {}
    for dd in day_dirs:
        sources = sorted(dd.glob("source_*.mp4"))
        if source_idx is not None:
            sources = [p for p in sources if p.stem == f"source_{source_idx}"]
        for src in sources:
            if src.stat().st_size == 0:
                log.warning("ghost: пустой файл %s, пропуск", src)
                continue
            out_dir = src.with_suffix("")
            for platform in platforms:
                try:
                    dst = uniquify(src, out_dir, platform, force=force, style=style)
                    result[f"{dd.name}/{src.stem}/{platform}"] = str(dst)
                except Exception as e:
                    log.error("ghost %s %s: %s", src, platform, e)
    return result


if __name__ == "__main__":
    from config_loader import cli_config
    from devices_registry import DevicesRegistry

    def extra(ap):
        ap.add_argument("--device", action="append", default=None)
        ap.add_argument("--day", default=None, help="YYYY-MM-DD (по умолчанию сегодня)")
        ap.add_argument("--all-days", action="store_true")
        ap.add_argument("--platforms", default=None)
        ap.add_argument("--force", action="store_true")

    cfg, a = cli_config("ghost", extra=extra)
    platforms = (a.platforms.split(",") if a.platforms
                 else cfg.uniqueness.get("platforms", ["tiktok", "instagram", "youtube"]))
    day = None if a.all_days else (a.day or cfg.today())
    registry = DevicesRegistry.from_config(cfg)
    cohorts = niche_cohorts(registry.get_active_devices(device_ids=None))
    out = {}
    for d in registry.get_active_devices(device_ids=a.device):
        out[d["device_id"]] = uniquify_device(d["device_id"], cfg.videos_root, platforms,
                                              day=day, force=a.force,
                                              cohort=cohorts.get(d.get("niche", "")),
                                              style_index=d.get("style_index"))
    sys.stdout.write(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
