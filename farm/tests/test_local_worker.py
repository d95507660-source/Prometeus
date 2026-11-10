"""tests/test_local_worker.py — local_worker, локальное хранилище model_server, переключение
cloud/local. torch, GPU и модели подменены моками; HTTP — FastAPI TestClient.

Запуск:  python -m pytest -q tests/test_local_worker.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config_loader  # noqa: E402
import local_worker as lw  # noqa: E402
import model_server  # noqa: E402
import run_generation_local as rgl  # noqa: E402

MB = 1024 * 1024
DAY = "2026-11-10"
SCENES = [{"prompt": "master reference portrait", "duration_sec": 0, "role": "master",
           "persona_lock": 777},
          {"prompt": "scene one", "duration_sec": 5, "reference": "master_reference", "role": "scene"}]


# ---------------------------------------------------------------------------
# Моки torch.cuda и конвейера
# ---------------------------------------------------------------------------

class OutOfMemoryError(RuntimeError):
    """То же имя класса, что у torch.cuda.OutOfMemoryError."""


class FakeCuda:
    def __init__(self, free_mb: int = 20000, total_mb: int = 24564, free_after_empty: int | None = None):
        self.free_mb = free_mb
        self.total_mb = total_mb
        self.free_after_empty = free_after_empty
        self.empty_calls = 0
        self.info_calls = 0

    def mem_get_info(self):
        self.info_calls += 1
        return self.free_mb * MB, self.total_mb * MB

    def empty_cache(self):
        self.empty_calls += 1
        if self.free_after_empty is not None:
            self.free_mb = self.free_after_empty


@dataclass
class FakePipeline:
    oom_above: int | None = None        # OOM, если max_frames больше
    fail_times: int = 0                 # сколько первых вызовов падают обычной ошибкой
    gate: threading.Event | None = None  # ждать этого события (для теста остановки)
    calls: list = field(default_factory=list)
    stores: int = 0
    running: threading.Event = field(default_factory=threading.Event)

    def __call__(self, payload, device_id, source_idx, voice_id, use_split, tmp, fetch, store):
        frames = payload["max_frames"]
        self.calls.append((device_id, source_idx, frames))
        self.running.set()
        if self.gate is not None:
            assert self.gate.wait(10)
        if self.fail_times > 0:
            self.fail_times -= 1
            raise RuntimeError("ffmpeg ошибка")
        if self.oom_above is not None and frames > self.oom_above:
            raise OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        master = tmp / "master_ref.png"
        if not fetch(master):
            master.write_bytes(b"png")
            store(master)
            self.stores += 1
        out = tmp / "final.mp4"
        out.write_bytes(f"{device_id}/{source_idx}/{frames}".encode())
        return out


def make_worker(tmp_path: Path, cuda: FakeCuda | None = None, pipeline: FakePipeline | None = None,
                **settings) -> tuple[lw.LocalWorker, dict]:
    counters = {"warm": 0}

    def warm():
        counters["warm"] += 1

    s = lw.WorkerSettings(**{"vram_wait_sec": 0.05, "heartbeat_sec": 0.2, **settings})
    w = lw.LocalWorker(tmp_path / "local_queue.db", tmp_path / "out", tmp_path / "models" / "masters", s,
                       pipeline=pipeline or FakePipeline(), warm=warm, cuda=cuda or FakeCuda(),
                       persona_seed=model_server._persona_seed, masters_io=model_server.masters_dir_io)
    return w, counters


def req(device="aria_fight", idx=1, **kw) -> dict:
    return {"device_id": device, "day": DAY, "source_idx": idx, "scenes": SCENES,
            "voice_id": "voice_01", "narration": "Hi", **kw}


def wait_for(cond, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError("условие не выполнилось за отведённое время")


@pytest.fixture
def started(tmp_path):
    workers = []

    def _make(**kw):
        w, c = make_worker(tmp_path, **kw)
        w.start()
        workers.append(w)
        return w, c

    yield _make
    for w in workers:
        w.shutdown(timeout=5)


# ---------------------------------------------------------------------------
# Воркер
# ---------------------------------------------------------------------------

def test_job_runs_once_models_loaded_once(started):
    pipe = FakePipeline()
    w, c = started(pipeline=pipe)
    ids = [w.submit(req(idx=i))["job_id"] for i in (1, 2, 3)]
    wait_for(lambda: all(w.status(i)["status"] == "done" for i in ids))
    assert c["warm"] == 1 and len(pipe.calls) == 3
    st = w.status(ids[0])
    assert st["output_ready"] and st["final"] and st["max_frames"] == 121 and st["attempts"] == 1
    assert w.result_path(ids[0]).read_bytes() == b"aria_fight/1/121"
    assert pipe.stores == 1  # опорный кадр: создан один раз, дальше из кэша
    assert (w.masters_dir / "aria_fight_777.png").exists()


def test_submit_is_idempotent(started):
    pipe = FakePipeline()
    w, _ = started(pipeline=pipe)
    a = w.submit(req())["job_id"]
    wait_for(lambda: w.status(a)["status"] == "done")
    again = w.submit(req())
    assert again == {"job_id": a, "status": "done"} and len(pipe.calls) == 1


def test_oom_retries_with_smaller_frames(started):
    pipe = FakePipeline(oom_above=49)
    w, _ = started(pipeline=pipe)
    j = w.submit(req())["job_id"]
    wait_for(lambda: w.status(j)["status"] == "done")
    assert [f for *_, f in pipe.calls] == [121, 81, 49]
    st = w.status(j)
    assert st["oom_count"] == 2 and st["max_frames"] == 49 and st["attempts"] == 3


def test_oom_on_last_step_is_final(started):
    pipe = FakePipeline(oom_above=10)
    w, _ = started(pipeline=pipe)
    j = w.submit(req())["job_id"]
    wait_for(lambda: w.status(j)["final"])
    time.sleep(0.2)
    st = w.status(j)
    assert st["status"] == "failed" and "OOM" in st["error"] and st["oom_count"] == 3
    assert [f for *_, f in pipe.calls] == [121, 81, 49]  # больше не повторяет


def test_failed_job_resubmitted_runs_again(started):
    pipe = FakePipeline(oom_above=10)
    w, _ = started(pipeline=pipe)
    j = w.submit(req())["job_id"]
    wait_for(lambda: w.status(j)["final"])
    pipe.oom_above = None
    assert w.submit(req()) == {"job_id": j, "status": "pending"}
    wait_for(lambda: w.status(j)["status"] == "done")
    assert w.status(j)["max_frames"] == 121  # новый payload — лимит с начала


def test_ordinary_error_retried(started):
    pipe = FakePipeline(fail_times=2)
    w, _ = started(pipeline=pipe)
    j = w.submit(req())["job_id"]
    wait_for(lambda: w.status(j)["status"] == "done")
    assert len(pipe.calls) == 3 and w.status(j)["oom_count"] == 0


def test_low_vram_waits_without_spending_attempts(tmp_path):
    cuda = FakeCuda(free_mb=1000)
    pipe = FakePipeline()
    w, _ = make_worker(tmp_path, cuda=cuda, pipeline=pipe)
    w.start()
    try:
        j = w.submit(req())["job_id"]
        wait_for(lambda: cuda.empty_calls >= 2)
        st = w.status(j)
        assert pipe.calls == [] and st["status"] == "pending" and st["attempts"] == 0
        cuda.free_mb = 8000  # память освободилась (другой процесс закрылся)
        wait_for(lambda: w.status(j)["status"] == "done")
        assert w.status(j)["attempts"] == 1
    finally:
        w.shutdown(timeout=5)


def test_empty_cache_can_free_enough(started):
    cuda = FakeCuda(free_mb=1500, free_after_empty=6000)
    pipe = FakePipeline()
    w, _ = started(cuda=cuda, pipeline=pipe)
    j = w.submit(req())["job_id"]
    wait_for(lambda: w.status(j)["status"] == "done")
    assert len(pipe.calls) == 1


def test_graceful_shutdown_waits_for_current_job(tmp_path):
    gate = threading.Event()
    pipe = FakePipeline(gate=gate)
    w, _ = make_worker(tmp_path, pipeline=pipe)
    w.start()
    first = w.submit(req(idx=1))["job_id"]
    second = w.submit(req(idx=2))["job_id"]
    assert pipe.running.wait(5)
    stopper = threading.Thread(target=w.shutdown)
    stopper.start()
    time.sleep(0.3)
    assert stopper.is_alive()  # ждёт текущую задачу
    with pytest.raises(RuntimeError):
        w.submit(req(idx=3))
    gate.set()
    stopper.join(5)
    assert not stopper.is_alive()
    assert w.status(first)["status"] == "done"
    assert w.status(second)["status"] == "pending" and w.status(second)["attempts"] == 0


def test_restart_recovers_interrupted_job(tmp_path):
    w, _ = make_worker(tmp_path)
    j = w.submit(req())["job_id"]
    w.q.register_worker("old-process")
    assert w.q.claim("old-process").id == j  # «процесс убит» посреди задачи
    w2, _ = make_worker(tmp_path)
    w2.start()
    try:
        wait_for(lambda: w2.status(j)["status"] == "done")
    finally:
        w2.shutdown(timeout=5)


def test_is_oom_detection():
    assert lw.is_oom(OutOfMemoryError("x"))
    assert lw.is_oom(RuntimeError("CUDA out of memory. Tried to allocate"))
    assert not lw.is_oom(RuntimeError("ffmpeg ошибка"))


# ---------------------------------------------------------------------------
# HTTP API
# ---------------------------------------------------------------------------

TOKEN = "worker-secret"


@pytest.fixture
def api(tmp_path):
    from fastapi.testclient import TestClient
    w, counters = make_worker(tmp_path)
    app = lw.create_app(w, TOKEN)
    with TestClient(app) as client:  # lifespan: start() при входе, shutdown() при выходе
        client.headers["Authorization"] = f"Bearer {TOKEN}"
        yield client, w, counters
    assert w.draining


def test_api_flow(api):
    client, w, counters = api
    assert counters["warm"] == 1
    r = client.post("/submit", json=req())
    assert r.status_code == 202
    j = r.json()["job_id"]
    wait_for(lambda: client.get(f"/status/{j}").json()["status"] == "done")
    res = client.get(f"/result/{j}")
    assert res.status_code == 200 and res.content == b"aria_fight/1/121"
    h = client.get("/health").json()
    assert h["ok"] and h["models_loaded"] and h["gpu"]["free_mb"] == 20000 and h["queue"]["done"] == 1


def test_api_auth_and_validation(api):
    client, w, _ = api
    assert client.get("/health", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/submit", json=req(device="../etc")).status_code == 422
    assert client.post("/submit", json=req(max_frames=2)).status_code == 422
    assert client.get("/status/999").status_code == 404
    w.draining = True
    assert client.post("/submit", json=req()).status_code == 503
    w.draining = False


def test_api_result_not_ready(tmp_path):
    from fastapi.testclient import TestClient
    gate = threading.Event()
    w, _ = make_worker(tmp_path, pipeline=FakePipeline(gate=gate))
    with TestClient(lw.create_app(w, None)) as client:
        j = client.post("/submit", json=req()).json()["job_id"]
        assert client.get(f"/result/{j}").status_code == 409
        gate.set()


# ---------------------------------------------------------------------------
# Клиент для forge (run_generation_local)
# ---------------------------------------------------------------------------

class ClientSession:
    """requests-подобная обёртка над TestClient: отбрасывает timeout."""

    def __init__(self, client):
        self.client = client

    @staticmethod
    def _path(url: str) -> str:
        from urllib.parse import urlsplit
        return urlsplit(url).path

    def post(self, url, json=None, headers=None, timeout=None):
        return self.client.post(self._path(url), json=json, headers=headers)

    def get(self, url, headers=None, timeout=None):
        return self.client.get(self._path(url), headers=headers)


def gen(session, out_path, **kw):
    return rgl.generate_for_device(device_id="aria_fight", scenes=SCENES, voice_id="voice_01",
                                   split_screen=False, source_idx=1, timeout_sec=60, day=DAY,
                                   narration="Hi", seed=42, negative_prompt="", out_path=out_path,
                                   server_url="http://worker:8081", poll_interval_sec=0.02,
                                   session=session, project_id="ignored", videos_root="ignored", **kw)


def test_client_end_to_end(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("LOCAL_WORKER_TOKEN", TOKEN)
    w, _ = make_worker(tmp_path)
    with TestClient(lw.create_app(w, TOKEN)) as client:
        out = tmp_path / "videos" / "aria_fight" / DAY / "source_1.mp4"
        res = gen(ClientSession(client), out)
    assert res["status"] == "ok" and res["execution"].startswith("local:")
    assert out.read_bytes() == b"aria_fight/1/121" and res["max_frames"] == 121
    assert isinstance(res["duration_sec"], float)


def test_client_reports_final_failure(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("LOCAL_WORKER_TOKEN", TOKEN)
    w, _ = make_worker(tmp_path, pipeline=FakePipeline(oom_above=10))
    with TestClient(lw.create_app(w, TOKEN)) as client:
        res = gen(ClientSession(client), tmp_path / "x.mp4")
    assert res["status"] == "failed" and "OOM" in res["error"]
    assert not (tmp_path / "x.mp4").exists()


def test_client_worker_unreachable(tmp_path):
    class Down:
        def post(self, *a, **kw):
            raise ConnectionError("connection refused")
    res = gen(Down(), tmp_path / "x.mp4")
    assert res["status"] == "failed" and "недоступен" in res["error"]


def test_client_rejects_bad_url(tmp_path):
    with pytest.raises(ValueError):
        rgl.generate_for_device(device_id="a", scenes=SCENES, voice_id="v", split_screen=False,
                                source_idx=1, timeout_sec=1, day=DAY, narration="", seed=1,
                                negative_prompt="", out_path=tmp_path / "x", server_url="file:///etc")


# ---------------------------------------------------------------------------
# model_server: STORAGE_BACKEND=local
# ---------------------------------------------------------------------------

@pytest.fixture
def local_storage(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("LOCAL_STORAGE_ROOT", str(tmp_path))

    def no_gcs():
        raise AssertionError("в локальном режиме google-cloud-storage не используется")

    monkeypatch.setattr(model_server, "_gcs", no_gcs)
    return tmp_path


def test_run_job_local_backend(local_storage, monkeypatch):
    root = local_storage
    (root / "in").mkdir()
    (root / "in" / "scenes.json").write_text(json.dumps({"scenes": SCENES, "narration": "Hi",
                                                         "master_ref_uri": "masters/aria.png"}),
                                             encoding="utf-8")
    seen = {}

    def fake_pipeline(payload, device_id, source_idx, voice_id, use_split, tmp, fetch, store):
        seen["fetch"] = fetch(tmp / "m.png")
        (tmp / "m.png").write_bytes(b"png")
        store(tmp / "m.png")
        out = tmp / "final.mp4"
        out.write_bytes(b"video")
        return out

    monkeypatch.setattr(model_server, "run_pipeline", fake_pipeline)
    monkeypatch.setenv("DEVICE_ID", "aria_fight")
    monkeypatch.setenv("SCENES_URI", "in/scenes.json")
    monkeypatch.setenv("OUTPUT_URI", f"file://{root}/out/aria_fight/source_1.mp4")
    res = model_server.run_job()
    assert res["storage"] == "local" and seen["fetch"] is False
    assert (root / "out" / "aria_fight" / "source_1.mp4").read_bytes() == b"video"
    assert (root / "masters" / "aria.png").read_bytes() == b"png"  # кэш опорного кадра локально


@pytest.mark.parametrize("uri", ["../outside.json", "/etc/passwd", "gs://bucket/key", "file:///etc/x"])
def test_local_backend_rejects_paths_outside_root(local_storage, uri):
    with pytest.raises(ValueError):
        model_server.local_path(uri)


def test_gcs_backend_requires_gs_uri(monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "gcs")
    with pytest.raises(ValueError):
        model_server.check_uri("/tmp/scenes.json")
    model_server.check_uri("gs://bucket/scenes.json")
    monkeypatch.setenv("STORAGE_BACKEND", "s3")
    with pytest.raises(ValueError):
        model_server.storage_backend()


def test_frames_limit_override():
    assert model_server._frames_for(5.0, 49) == 49
    assert model_server._frames_for(5.0, 81) == 81
    assert model_server._frames_for(5.0) == min(121, model_server.WAN_MAX_FRAMES)
    assert model_server._payload_max_frames({"max_frames": 49}) == 49
    assert model_server._payload_max_frames({}) is None
    with pytest.raises(ValueError):
        model_server._payload_max_frames({"max_frames": True})


# ---------------------------------------------------------------------------
# Переключение cloud/local в forge, forge_stage и конфиге
# ---------------------------------------------------------------------------

def test_forge_picks_local_backend(monkeypatch):
    import forge
    f = forge.Forge.__new__(forge.Forge)
    f.cfg = SimpleNamespace(gcp=SimpleNamespace(mode="local"),
                            local=config_loader.LocalConfig(server_url="http://127.0.0.1:8081"))
    fn, extra = f._generator()
    assert fn is rgl.generate_for_device
    assert extra == {"server_url": "http://127.0.0.1:8081", "poll_interval_sec": 5.0}
    sentinel = SimpleNamespace(generate_for_device=object())
    monkeypatch.setitem(sys.modules, "run_generation_gcp", sentinel)
    f.cfg = SimpleNamespace(gcp=SimpleNamespace(mode="cloud"))
    assert f._generator() == (sentinel.generate_for_device, {})


def test_forge_stage_local_mode_skips_cloud_budget(tmp_path, monkeypatch):
    import budget_guard
    import forge_stage

    def boom(cfg):
        raise AssertionError("облачный бюджет не должен включаться в local")

    monkeypatch.setattr(budget_guard, "guard_for", boom)
    cfg = SimpleNamespace(gcp=SimpleNamespace(mode="local"),
                          budget=config_loader.BudgetConfig(enabled=True),
                          forge=config_loader.ForgeConfig(videos_per_device_per_day=1),
                          queue_db_path=tmp_path / "q.db", today=lambda: DAY)

    class Reg:
        def get_active_devices(self, niches=None, device_ids=None):
            return [{"device_id": "aria_fight", "niche": "n", "accounts": {}}]

    class F:
        def generate_one(self, d, i, day):
            return SimpleNamespace(status="ok", error=None, gcp_execution_name="local:1")

    code = forge_stage.stage_forge(cfg, Reg(), forge=F(), settings=forge_stage.RunSettings(
        max_workers=1, poll_sec=0.01, heartbeat_sec=0.2))
    assert code == 0


def _write_cfg(tmp_path: Path, mutate=None) -> Path:
    raw = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    if mutate:
        mutate(raw)
    (tmp_path / "config.json").write_text(json.dumps(raw), encoding="utf-8")
    niches = {n: {"marker": m} for n, m in raw["devices_registry"]["niche_markers"].items()}
    (tmp_path / "niches.json").write_text(json.dumps({"niches": niches}), encoding="utf-8")
    return tmp_path / "config.json"


def test_config_local_block(tmp_path, caplog, monkeypatch):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    cfg = config_loader.load_config(_write_cfg(tmp_path))
    assert cfg.gcp.mode == "cloud" and cfg.local.oom_max_frames == [81, 49]
    assert cfg.local_queue_db_path == tmp_path / "data" / "local_queue.db"

    def to_local(r):
        r["gcp"]["mode"] = "local"
        r["gcp"].pop("project_id")
    cfg = config_loader.load_config(_write_cfg(tmp_path, to_local))  # project_id не нужен
    assert cfg.gcp.mode == "local"


@pytest.mark.parametrize("mutate, msg", [
    (lambda r: r["gcp"].update(mode="hybrid"), "gcp.mode"),
    (lambda r: r["gcp"].pop("project_id"), "project_id"),
    (lambda r: r["local"].update(oom_max_frames=[49, 81]), "oom_max_frames"),
    (lambda r: r["local"].update(oom_max_frames=[80]), "oom_max_frames"),
    (lambda r: r["local"].update(server_url="ftp://x"), "server_url"),
    (lambda r: r["local"].update(max_concurrent_jobs=0), "max_concurrent_jobs"),
    (lambda r: r["local"].update(vram_safety_mb=-1), "vram_safety_mb"),
])
def test_config_local_validation(tmp_path, monkeypatch, mutate, msg):
    monkeypatch.delenv("GCP_PROJECT_ID", raising=False)
    with pytest.raises(ValueError, match=msg):
        config_loader.load_config(_write_cfg(tmp_path, mutate))
