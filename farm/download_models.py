"""download_models.py — синхронизация моделей из GCS в контейнер (вызывается из download_models.sh)."""
from __future__ import annotations

import argparse
import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

log = logging.getLogger("download_models")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--dest", default="/models")
    ap.add_argument("--prefix", default="models/")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    import importlib
    storage = importlib.import_module("google.cloud.storage")  # тип Any — без зависимости от стабов
    client = storage.Client()
    dest = Path(a.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    todo = []
    for blob in client.list_blobs(a.bucket, prefix=a.prefix):
        rel = blob.name[len(a.prefix):]
        if not rel or rel.endswith("/"):
            continue
        dst = (dest / rel).resolve()
        if dest not in dst.parents:
            log.warning("пропуск подозрительного объекта %s", blob.name)
            continue
        if dst.exists() and blob.size is not None and dst.stat().st_size == blob.size:
            continue
        todo.append((blob, dst))

    def fetch(item):
        blob, dst = item
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        blob.download_to_filename(str(tmp), timeout=3600)
        tmp.replace(dst)
        return blob.name

    errors = 0
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        for fut in [ex.submit(fetch, t) for t in todo]:
            try:
                log.info("ok %s", fut.result())
            except Exception as e:
                errors += 1
                log.error("ошибка: %s", e)
    log.info("скачано %d, ошибок %d", len(todo) - errors, errors)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
