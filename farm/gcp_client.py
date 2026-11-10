"""gcp_client.py — клиент GCP: Cloud Storage + запуск Cloud Run Job и ожидание выполнения."""
from __future__ import annotations

import importlib
import json
import logging
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Необязательные GCP-библиотеки грузим через importlib: локально (без GCP) модуль импортируется,
# а mypy видит их одинаково (Any) независимо от того, какие из пакетов google-cloud-* установлены.
storage: Any = None
run_v2: Any = None
_IMPORT_ERROR: Exception | None = None
try:
    storage = importlib.import_module("google.cloud.storage")
    run_v2 = importlib.import_module("google.cloud.run_v2")
except ImportError as e:  # локальный запуск без GCP-зависимостей
    storage = run_v2 = None
    _IMPORT_ERROR = e


def parse_gs_uri(uri: str) -> tuple[str, str]:
    """gs://bucket/key → (bucket, key). Без assert: он отключается при python -O."""
    if not isinstance(uri, str) or not uri.startswith("gs://"):
        raise ValueError(f"ожидался gs:// URI, получено: {uri!r}")
    rest = uri[5:]
    if "/" not in rest:
        raise ValueError(f"в URI нет пути объекта: {uri!r}")
    bucket, key = rest.split("/", 1)
    if not bucket or not key or ".." in key.split("/"):
        raise ValueError(f"некорректный URI: {uri!r}")
    return bucket, key


class GcpClient:
    def __init__(self, project_id: str, region: str = "us-central1",
                 job_name: str = "prometheus-generate",
                 work_bucket: str = "prometheus-work", models_bucket: str = "prometheus-models"):
        if storage is None or run_v2 is None:
            raise RuntimeError(f"установи google-cloud-storage и google-cloud-run ({_IMPORT_ERROR})")
        if not project_id:
            raise ValueError("project_id пустой")
        self.project_id = project_id
        self.region = region
        self.job_name = job_name
        self.work_bucket = work_bucket
        self.models_bucket = models_bucket
        self._storage = storage.Client(project=project_id)
        self._jobs = run_v2.JobsClient()
        self._executions = run_v2.ExecutionsClient()  # get/list execution живут здесь, не в JobsClient

    # ------------------------------------------------------------------ Storage
    def _blob(self, gcs_uri: str):
        bucket_name, key = parse_gs_uri(gcs_uri)
        return self._storage.bucket(bucket_name).blob(key)

    def upload_file(self, local_path: str | Path, gcs_uri: str) -> str:
        self._blob(gcs_uri).upload_from_filename(str(local_path), timeout=600)
        log.info("uploaded %s -> %s", local_path, gcs_uri)
        return gcs_uri

    def download_file(self, gcs_uri: str, local_path: str | Path) -> Path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = local_path.with_suffix(local_path.suffix + ".part")
        self._blob(gcs_uri).download_to_filename(str(tmp), timeout=600)
        tmp.replace(local_path)  # без «полускачанных» файлов
        log.info("downloaded %s -> %s", gcs_uri, local_path)
        return local_path

    def exists(self, gcs_uri: str) -> bool:
        return self._blob(gcs_uri).exists()

    def upload_json(self, data: dict, gcs_uri: str) -> str:
        self._blob(gcs_uri).upload_from_string(
            json.dumps(data, ensure_ascii=False), content_type="application/json", timeout=120)
        return gcs_uri

    def upload_stock_clip(self, local_path: str | Path, niche: str, clip_id: str) -> str:
        return self.upload_file(local_path,
                                f"gs://{self.models_bucket}/models/stock/{niche}/{clip_id}.mp4")

    def download_models(self, models_dir: str | Path = "/models", prefix: str = "models/") -> int:
        models_dir = Path(models_dir).resolve()
        models_dir.mkdir(parents=True, exist_ok=True)
        n = 0
        for blob in self._storage.list_blobs(self.models_bucket, prefix=prefix):
            rel = blob.name[len(prefix):]
            if not rel or rel.endswith("/"):  # «папки»-заглушки
                continue
            dst = (models_dir / rel).resolve()
            if models_dir not in dst.parents:  # защита от ../ в именах объектов
                log.warning("пропускаю подозрительный объект: %s", blob.name)
                continue
            if dst.exists() and blob.size is not None and dst.stat().st_size == blob.size:
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(dst), timeout=1800)
            n += 1
        log.info("скачано файлов моделей: %d -> %s", n, models_dir)
        return n

    # ------------------------------------------------------------------ Cloud Run Job
    def _job_path(self) -> str:
        return f"projects/{self.project_id}/locations/{self.region}/jobs/{self.job_name}"

    def run_job(self, env_overrides: dict[str, str], timeout_sec: int = 3600) -> dict:
        overrides = run_v2.RunJobRequest.Overrides(
            container_overrides=[
                run_v2.RunJobRequest.Overrides.ContainerOverride(
                    env=[run_v2.EnvVar(name=k, value=str(v)) for k, v in env_overrides.items()]
                )
            ],
            timeout=timedelta(seconds=timeout_sec),
        )
        req = run_v2.RunJobRequest(name=self._job_path(), overrides=overrides)
        op = self._jobs.run_job(request=req)

        execution_name = None
        try:
            md = op.metadata  # для run_job метаданные операции — это Execution
            execution_name = getattr(md, "name", None) or None
        except Exception:
            execution_name = None
        log.info("Cloud Run Job запущен: %s execution=%s", self._job_path(), execution_name)

        if execution_name:
            return self.wait_execution(execution_name, timeout_sec=timeout_sec + 300)

        # запасной путь: ждём завершения операции, она завершается вместе с выполнением
        try:
            ex = op.result(timeout=timeout_sec + 300)
            # операция завершена, но статус может не определиться — не возвращаем None вызывающему
            return self._execution_status(ex) or {"status": "UNKNOWN", "execution": ex.name}
        except Exception as e:
            log.error("job failed: %s", e)
            return {"status": "FAILED", "error": str(e)}

    @staticmethod
    def _execution_status(ex) -> dict | None:
        """None — ещё выполняется; иначе {'status': SUCCEEDED|FAILED|CANCELLED, ...}.

        У Execution нет поля terminal_condition (оно есть у Job): проверка `"terminal_condition"
        in ex` бросала AttributeError, wait_execution принимал это за сбой сети, ждал до таймаута
        и отменял даже успешное выполнение. Итог берём из условия "Completed" в ex.conditions
        [ПРОВЕРИТЬ ПО ДОКАМ: тип условия], а если его нет — по completion_time и счётчикам задач.
        """
        Cond = run_v2.Condition
        tc = next((c for c in (getattr(ex, "conditions", None) or [])
                   if getattr(c, "type_", "") == "Completed"), None)
        state = tc.state if tc is not None else None
        done = "completion_time" in ex
        info = {"execution": ex.name,
                "succeeded": int(getattr(ex, "succeeded_count", 0) or 0),
                "failed": int(getattr(ex, "failed_count", 0) or 0),
                "cancelled": int(getattr(ex, "cancelled_count", 0) or 0)}
        # длительность выполнения для учёта бюджета (budget_guard): completion_time - start_time
        # [ПРОВЕРИТЬ ПО ДОКАМ: поля Execution в google-cloud-run; при task_count > 1 GPU-время
        #  больше длительности выполнения]
        try:
            if "start_time" in ex and "completion_time" in ex:
                info["duration_sec"] = max(0.0, (ex.completion_time - ex.start_time).total_seconds())
        except Exception:
            pass
        if state == Cond.State.CONDITION_SUCCEEDED:
            return {"status": "SUCCEEDED", **info}
        if state == Cond.State.CONDITION_FAILED:
            status = "CANCELLED" if info["cancelled"] else "FAILED"
            return {"status": status, "reason": getattr(tc, "message", ""), **info}
        if done and not getattr(ex, "reconciling", False):
            task_count = int(getattr(ex, "task_count", 1) or 1)
            if info["succeeded"] >= task_count:
                return {"status": "SUCCEEDED", **info}
            return {"status": "CANCELLED" if info["cancelled"] else "FAILED", **info}
        return None

    def wait_execution(self, execution_name: str, timeout_sec: int = 3600,
                       poll_sec: float = 15.0) -> dict:
        deadline = time.time() + timeout_sec
        delay = 5.0
        while time.time() < deadline:
            try:
                ex = self._executions.get_execution(name=execution_name)
                st = self._execution_status(ex)
                if st is not None:
                    log.info("execution %s → %s", execution_name, st["status"])
                    return st
            except Exception as e:  # сетевые сбои при опросе не валят ожидание
                log.warning("get_execution %s: %s", execution_name, e)
            time.sleep(delay)
            delay = min(poll_sec, delay * 1.5)
        try:
            self._executions.cancel_execution(name=execution_name)
            log.warning("execution %s отменён по таймауту", execution_name)
        except Exception:
            pass
        return {"status": "TIMEOUT", "execution": execution_name}
