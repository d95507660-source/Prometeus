"""run_generation_local.py — generate_for_device для gcp.mode=local: отправка задачи в local_worker.

Та же сигнатура, что у run_generation_gcp.generate_for_device (лишние аргументы игнорируются),
поэтому forge и forge_stage работают без изменений: forge выбирает модуль по cfg.gcp.mode.

Ожидание: пока задача стоит в очереди воркера — до queue_timeout_sec (на один GPU очередь из
десятков роликов — это часы); с момента старта генерации — не дольше timeout_sec.
Токен — LOCAL_WORKER_TOKEN (тот же, что у воркера).
"""
from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

URL_RE = re.compile(r"^https?://[A-Za-z0-9.\-]{1,253}(:\d{1,5})?/?$")
DEFAULT_QUEUE_TIMEOUT_SEC = 12 * 3600


class LocalWorkerError(RuntimeError):
    pass


def _request(fn: Any, url: str, what: str, **kw: Any) -> Any:
    """HTTP-вызов: сетевые ошибки (requests.ConnectionError — тоже OSError) отделяем от ошибок
    записи файла, чтобы в логе было видно, что воркер просто не запущен."""
    try:
        resp = fn(url, **kw)
    except Exception as e:
        raise LocalWorkerError(f"local_worker недоступен ({what}): {type(e).__name__}: {e}") from e
    return _check(resp, what)


def _check(resp: Any, what: str) -> Any:
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail")
        except Exception:
            detail = ""
        raise LocalWorkerError(f"local_worker {what}: HTTP {resp.status_code} {detail}".strip())
    return resp


def generate_for_device(*, device_id: str, scenes: list[dict], voice_id: str, split_screen: bool,
                        source_idx: int, timeout_sec: int, day: str, narration: str, seed: int,
                        negative_prompt: str, out_path: Path, server_url: str,
                        poll_interval_sec: float = 5.0,
                        queue_timeout_sec: float = DEFAULT_QUEUE_TIMEOUT_SEC,
                        session: Any = None, **_ignored: Any) -> dict:
    if not URL_RE.fullmatch(server_url or ""):
        raise ValueError(f"некорректный server_url: {server_url!r}")
    base = server_url.rstrip("/")
    if session is None:
        import requests
        session = requests.Session()
    token = os.getenv("LOCAL_WORKER_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    persona_lock = next((s.get("persona_lock") for s in scenes
                         if isinstance(s, dict) and s.get("role") == "master"), None)
    body = {"device_id": device_id, "day": day, "source_idx": int(source_idx), "scenes": scenes,
            "voice_id": voice_id, "split_screen": bool(split_screen), "narration": narration or "",
            "seed": seed, "negative_prompt": negative_prompt or "", "persona_lock": persona_lock}
    body = {k: v for k, v in body.items() if v is not None}

    try:
        job = _request(session.post, f"{base}/submit", "submit", json=body, headers=headers,
                       timeout=30).json()
        job_id = int(job["job_id"])
        log.info("local_worker: %s/%s/source_%d → задача %d", device_id, day, source_idx, job_id)
        submitted = time.monotonic()
        while True:
            st = _request(session.get, f"{base}/status/{job_id}", "status", headers=headers,
                          timeout=30).json()
            if st["status"] == "done":
                break
            if st["final"]:
                return {"status": "failed", "error": st.get("error") or "задача упала",
                        "execution": f"local:{job_id}"}
            if st["status"] == "running" and st.get("started_at"):
                if time.time() - float(st["started_at"]) > timeout_sec:
                    return {"status": "timeout", "execution": f"local:{job_id}",
                            "error": f"генерация дольше {timeout_sec} с"}
            elif time.monotonic() - submitted > queue_timeout_sec:
                return {"status": "timeout", "execution": f"local:{job_id}",
                        "error": f"в очереди воркера дольше {queue_timeout_sec:.0f} с"}
            time.sleep(poll_interval_sec)

        resp = _request(session.get, f"{base}/result/{job_id}", "result", headers=headers, timeout=300)
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = out_path.with_name(f".{out_path.name}.part")
        try:
            tmp.write_bytes(resp.content)
            os.replace(tmp, out_path)
        finally:
            tmp.unlink(missing_ok=True)
    except LocalWorkerError as e:
        return {"status": "failed", "error": str(e)[:500]}
    except (OSError, ValueError, KeyError) as e:  # запись файла, неожиданный ответ воркера
        return {"status": "failed", "error": f"{type(e).__name__}: {e}"[:500]}
    return {"status": "ok", "local_path": str(out_path), "execution": f"local:{job_id}",
            "duration_sec": st.get("run_sec"), "max_frames": st.get("max_frames")}
