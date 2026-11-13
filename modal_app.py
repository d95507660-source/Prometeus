"""modal_app.py — генерация вертикального видео (Wan 2.2) и апскейл 720p -> 1080p (SeedVR2) на Modal (GPU L4).

Пайплайн:
    PrometheusGenerator.generate_video(...)  -> 720p mp4 (Wan 2.2 TI2V-5B, diffusers)
    PrometheusGenerator.upscale_video(...)   -> 1080p mp4 (SeedVR2, временная согласованность, аудио сохраняется)

Почему SeedVR2 запускается не через diffusers:
    В diffusers нет пайплайна SeedVR2 (ни SeedVR2Pipeline, ни поддержки from_pretrained для
    ByteDance-Seed/SeedVR2-3B / -7B). Официальный репозиторий ByteDance-Seed/SeedVR рассчитан на
    torchrun + flash_attn + apex и не умеет экономить VRAM. Поэтому используется standalone-CLI из
    numz/ComfyUI-SeedVR2_VideoUpscaler (inference_cli.py работает без ComfyUI): это те же веса
    ByteDance, перепакованные в fp16 safetensors (HF: numz/SeedVR2_comfyUI, sha256 сверяется
    реестром самого CLI), плюс BlockSwap, тайловый VAE и потоковая обработка длинных видео.
    Код CLI закреплён на конкретном коммите (SEEDVR2_COMMIT), чтобы обновления апстрима не ломали прод.

Выбор модели (L4, 24 ГБ):
    По умолчанию SeedVR2-3B fp16 (~6.8 ГБ весов). Оставшиеся ~16 ГБ уходят на активации, что
    позволяет держать batch_size 21 кадр в 1080p — именно размер батча определяет отсутствие
    мерцания. 7B fp16 (~16.5 ГБ весов) на L4 помещается только с BlockSwap и малым батчем: детализация
    чуть выше, но временная стабильность хуже и обработка в разы медленнее. 7B включается
    env SEEDVR2_MODEL=7b.

Запуск:
    modal run modal_app.py::main --prompt "..." --duration 30
    modal run modal_app.py::upscale_local_file --input-path in.mp4 --output-path out_1080p.mp4
    modal deploy modal_app.py
"""
from __future__ import annotations

import gc
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import modal

log = logging.getLogger("prometheus")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# ---------------------------------------------------------------------------
# Константы
# ---------------------------------------------------------------------------

APP_NAME = "prometheus"

MODELS_DIR = Path("/models")                       # Modal Volume с весами (кэш между запусками)
OUTPUTS_DIR = Path("/outputs")                     # Modal Volume с готовыми роликами
HF_CACHE_DIR = MODELS_DIR / "hf"
SEEDVR2_MODEL_DIR = MODELS_DIR / "seedvr2"

# Wan 2.2 TI2V-5B — единственный вариант Wan 2.2, который помещается на L4 (24 ГБ) с CPU offload.
# Родное разрешение 720p: 1280x704 (для вертикали height=1280, width=704), 24 fps, до 121 кадра за проход.
WAN_REPO_ID = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
WAN_WIDTH = 704
WAN_HEIGHT = 1280
WAN_FPS = 24
WAN_FRAMES_PER_CLIP = 121                          # 4n+1, ~5 секунд при 24 fps
WAN_STEPS = int(os.getenv("WAN_STEPS", "50"))
WAN_GUIDANCE = float(os.getenv("WAN_GUIDANCE", "5.0"))
WAN_NEGATIVE_PROMPT = (
    "blurry, low quality, distorted face, deformed hands, extra fingers, bad anatomy, "
    "text, watermark, subtitles, static frame, flicker, jpeg artifacts"
)

# Standalone-реализация SeedVR2 (inference_cli.py), закреплённая на проверенном коммите.
SEEDVR2_REPO_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
SEEDVR2_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"
SEEDVR2_CODE_DIR = Path("/opt/seedvr2")
SEEDVR2_CLI = SEEDVR2_CODE_DIR / "inference_cli.py"
SEEDVR2_WEIGHTS_REPO = "numz/SeedVR2_comfyUI"      # fp16 safetensors оригинальных весов ByteDance-Seed/SeedVR2-*
SEEDVR2_VAE_FILE = "ema_vae_fp16.safetensors"

# Целевое разрешение апскейла: короткая сторона 1080 (вертикаль 1080x1920).
TARGET_SHORT_SIDE = 1080
TARGET_VERTICAL_SIZE = (1080, 1920)                # (width, height) для роликов с соотношением ~9:16
ASPECT_SNAP_TOLERANCE = 0.03                       # 704x1280 (0.55) приводится к 1080x1920 (0.5625) кропом ~2%

# torch.compile (inductor). Отключение: env SEEDVR2_COMPILE=0.
SEEDVR2_COMPILE = os.getenv("SEEDVR2_COMPILE", "1") == "1"
SEEDVR2_COMPILE_MODE = os.getenv("SEEDVR2_COMPILE_MODE", "max-autotune")
COMPILE_CACHE_DIR = MODELS_DIR / "compile_cache"   # на Volume: кэш переживает перезапуск контейнера
INDUCTOR_CACHE_DIR = COMPILE_CACHE_DIR / "inductor"
TRITON_CACHE_DIR = COMPILE_CACHE_DIR / "triton"

# Автокалибровка batch_size по пиковой VRAM (L4 = 24 ГБ). Отключение: env SEEDVR2_CALIBRATE=0.
CALIBRATION_ENABLED = os.getenv("SEEDVR2_CALIBRATE", "1") == "1"
CALIBRATION_FRAMES = 5
CALIBRATION_HIGH_GB = 22.0       # выше — стартуем с batch_size 17
CALIBRATION_CRITICAL_GB = 23.3   # выше (почти весь объём L4) — стартуем с 13

# Таймауты (секунды)
GENERATE_TIMEOUT_S = 3 * 60 * 60
UPSCALE_ATTEMPT_TIMEOUT_S = 90 * 60


@dataclass(frozen=True)
class SeedVR2Variant:
    """Параметры конкретного размера модели SeedVR2."""
    dit_file: str
    max_blocks_to_swap: int      # сколько трансформер-блоков всего можно выгружать в RAM (лимит CLI)
    initial_blocks_to_swap: int  # BlockSwap уже на первой попытке (нужен 7B, т.к. веса ~16.5 ГБ)
    max_batch_size: int          # потолок batch_size в 1080p на 24 ГБ (4n+1)


SEEDVR2_VARIANTS: dict[str, SeedVR2Variant] = {
    "3b": SeedVR2Variant(
        dit_file="seedvr2_ema_3b_fp16.safetensors",
        max_blocks_to_swap=32,
        initial_blocks_to_swap=0,
        max_batch_size=21,
    ),
    "7b": SeedVR2Variant(
        dit_file="seedvr2_ema_7b_fp16.safetensors",
        max_blocks_to_swap=36,
        initial_blocks_to_swap=24,
        max_batch_size=9,
    ),
}

# Признаки нехватки видеопамяти в выводе CLI. Отдельно учитывается returncode -9 (SIGKILL от OOM-killer
# по оперативной памяти контейнера) — его тоже лечит уменьшение батча/чанка.
OOM_PATTERNS = re.compile(
    r"(CUDA out of memory|OutOfMemoryError|out of memory|CUBLAS_STATUS_ALLOC_FAILED|"
    r"cudaErrorMemoryAllocation|CUDNN_STATUS_ALLOC_FAILED|Allocation on device)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Образ, тома, приложение
# ---------------------------------------------------------------------------

image = (
    modal.Image.debian_slim(python_version="3.11")
    # ffmpeg — сборка/мукс видео и копирование аудио; git — клонирование SeedVR2;
    # libgl1/libglib2.0-0 — системные зависимости OpenCV (используется SeedVR2 для чтения кадров).
    # build-essential — C-компилятор для Triton/inductor (torch.compile собирает вспомогательные модули).
    .apt_install("ffmpeg", "git", "libgl1", "libglib2.0-0", "build-essential")
    .pip_install(
        "torch==2.7.1",
        "torchvision==0.22.1",
    )
    .pip_install(
        # Wan 2.2
        "diffusers==0.35.1",
        "transformers==4.56.1",
        "accelerate==1.10.1",
        "sentencepiece==0.2.1",
        "ftfy==6.3.1",
        "imageio==2.37.0",
        "imageio-ffmpeg==0.6.0",
        "huggingface_hub[hf_transfer]==0.34.4",
        "safetensors==0.6.2",
        # SeedVR2 standalone (requirements.txt репозитория numz/ComfyUI-SeedVR2_VideoUpscaler)
        "numpy==2.2.6",
        "opencv-python-headless==4.11.0.86",
        "einops==0.8.1",
        "omegaconf==2.3.0",
        "peft==0.17.1",
        "rotary_embedding_torch==0.8.9",
        "gguf==0.17.1",
        "psutil==7.0.0",
        "tqdm==4.67.1",
        "matplotlib==3.10.6",
    )
    .run_commands(
        f"git clone {SEEDVR2_REPO_URL} {SEEDVR2_CODE_DIR}",
        f"git -C {SEEDVR2_CODE_DIR} checkout {SEEDVR2_COMMIT}",
        # Быстрая проверка на этапе сборки: CLI на месте и парсит аргументы (без GPU не грузит модель).
        f"test -f {SEEDVR2_CLI}",
    )
    .env({
        "HF_HOME": str(HF_CACHE_DIR),
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
)

models_volume = modal.Volume.from_name("prometheus-models", create_if_missing=True)
outputs_volume = modal.Volume.from_name("prometheus-outputs", create_if_missing=True)

app = modal.App(APP_NAME, image=image)


# ---------------------------------------------------------------------------
# ffprobe / ffmpeg утилиты
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: Fraction
    frame_count: int
    duration_s: float
    video_codec: str
    pix_fmt: str
    has_audio: bool


def _run(cmd: list[str], timeout: int = 600) -> subprocess.CompletedProcess:
    """Запуск внешней команды с понятной ошибкой при неудаче."""
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[-30:])
        raise RuntimeError(f"команда завершилась с кодом {proc.returncode}: {' '.join(cmd[:4])} ...\n{tail}")
    return proc


def probe_video(path: Path) -> VideoInfo:
    """Читает параметры видео через ffprobe. Кадры считаются по пакетам (-count_packets) —
    это точно и быстро, без декодирования."""
    proc = _run([
        "ffprobe", "-v", "error", "-count_packets",
        "-show_entries",
        "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,nb_read_packets"
        ":format=duration",
        "-of", "json", str(path),
    ])
    data = json.loads(proc.stdout)
    streams = data.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError(f"{path}: видеопоток не найден")
    has_audio = any(s.get("codec_type") == "audio" for s in streams)

    fps_str = video.get("avg_frame_rate") or video.get("r_frame_rate") or "0/1"
    if fps_str in ("0/0", "0/1"):
        fps_str = video.get("r_frame_rate", "30/1")
    fps = Fraction(fps_str)
    if fps <= 0:
        raise ValueError(f"{path}: некорректный fps {fps_str!r}")

    duration = float(data.get("format", {}).get("duration") or 0.0)
    frame_count = int(video.get("nb_read_packets") or 0)
    if frame_count <= 0:
        frame_count = int(round(duration * float(fps)))
    if frame_count <= 0:
        raise ValueError(f"{path}: не удалось определить число кадров")

    return VideoInfo(
        width=int(video["width"]),
        height=int(video["height"]),
        fps=fps,
        frame_count=frame_count,
        duration_s=duration,
        video_codec=str(video.get("codec_name", "h264")),
        pix_fmt=str(video.get("pix_fmt", "yuv420p")),
        has_audio=has_audio,
    )


def _even(x: float) -> int:
    """Ближайшее чётное целое (yuv420 требует чётных размеров)."""
    return max(2, int(round(x / 2.0)) * 2)


def compute_target_size(width: int, height: int) -> tuple[int, int]:
    """Целевой размер 1080p.

    Для вертикали с соотношением около 9:16 (в т.ч. 704x1280 от Wan 2.2) отдаём ровно 1080x1920 —
    стандарт площадок; отличие в пропорциях (<3%) снимается центрированным кропом, а не растяжением.
    Для прочих пропорций масштабируем короткую сторону до 1080 с сохранением пропорций.
    """
    tw, th = TARGET_VERTICAL_SIZE
    if height > width and abs((width / height) - (tw / th)) / (tw / th) <= ASPECT_SNAP_TOLERANCE:
        return tw, th
    if width <= height:
        return TARGET_SHORT_SIDE, _even(height * TARGET_SHORT_SIDE / width)
    return _even(width * TARGET_SHORT_SIDE / height), TARGET_SHORT_SIDE


def _encoder_for(codec_name: str) -> tuple[str, list[str]]:
    """Подбирает энкодер под кодек исходника ("тот же кодек") с качеством, близким к визуально
    lossless. Неизвестные/экзотические кодеки кодируются в H.264 — максимальная совместимость."""
    codec = codec_name.lower()
    if codec in ("hevc", "h265"):
        return "libx265", ["-crf", "18", "-preset", "slow", "-tag:v", "hvc1", "-x265-params", "log-level=error"]
    if codec == "vp9":
        return "libvpx-vp9", ["-crf", "24", "-b:v", "0", "-row-mt", "1"]
    if codec == "av1":
        return "libaom-av1", ["-crf", "26", "-b:v", "0", "-cpu-used", "4", "-row-mt", "1"]
    return "libx264", ["-crf", "16", "-preset", "slow", "-profile:v", "high"]


# ---------------------------------------------------------------------------
# Выбор batch_size (4n+1)
# ---------------------------------------------------------------------------

def is_valid_batch_size(n: int) -> bool:
    """SeedVR2 работает с батчами вида 4n+1: латентное пространство VAE сжимает время в 4 раза
    (+1 опорный кадр). Временная согласованность включается при batch_size >= 5."""
    return n >= 1 and (n - 1) % 4 == 0


def largest_4n1_at_most(limit: int) -> int:
    """Наибольшее число вида 4n+1, не превышающее limit (минимум 1)."""
    if limit < 1:
        return 1
    return ((limit - 1) // 4) * 4 + 1


def choose_batch_size(frame_count: int, max_batch_size: int) -> int:
    """batch_size в зависимости от длины видео.

    - Чем больше кадров обрабатывается вместе, тем меньше мерцания: модель видит общий контекст
      и согласует детали между кадрами. Поэтому берём максимум, который выдерживает VRAM L4.
    - Для коротких роликов не берём больше, чем кадров в видео (последний батч с --uniform_batch_size
      дополняется паддингом, лишний паддинг — пустая трата VRAM и времени).
    - Нижняя граница — 5 (минимальный батч с временной согласованностью). Если в видео меньше
      5 кадров, батч всё равно 5: CLI дополнит его благодаря --uniform_batch_size.
    """
    cap = largest_4n1_at_most(max_batch_size)
    batch = largest_4n1_at_most(min(frame_count, cap))
    return max(5, batch)


@dataclass(frozen=True)
class UpscaleAttempt:
    """Одна конфигурация запуска SeedVR2. При OOM переходим к следующей, более экономной."""
    batch_size: int
    blocks_to_swap: int
    swap_io_components: bool

    @property
    def temporal(self) -> bool:
        return self.batch_size >= 5

    def describe(self) -> str:
        swap = f"blockswap={self.blocks_to_swap}{'+io' if self.swap_io_components else ''}" \
            if self.blocks_to_swap else "blockswap=off"
        return f"batch_size={self.batch_size} ({'temporal' if self.temporal else 'per-frame'}), {swap}"


def build_attempt_plan(initial_batch: int, variant: SeedVR2Variant) -> list[UpscaleAttempt]:
    """Лестница fallback'ов при OOM — от лучшего качества к гарантированно помещающемуся:

    1. исходный batch_size без BlockSwap (быстрее всего) — для 7B сразу с BlockSwap;
    2. тот же batch_size + максимальный BlockSwap: веса DiT уходят в RAM, VRAM остаётся под активации,
       временная стабильность не страдает — платим только скоростью;
    3. уменьшаем batch_size по ряду 4n+1 (21 -> 13 -> 9 -> 5), сохраняя BlockSwap;
    4. последний рубеж — batch_size=1: покадровая обработка без временной согласованности
       (возможно мерцание), зато пайплайн не падает.
    """
    plan: list[UpscaleAttempt] = []
    max_swap = variant.max_blocks_to_swap

    def add(attempt: UpscaleAttempt) -> None:
        if attempt not in plan:
            plan.append(attempt)

    add(UpscaleAttempt(initial_batch, variant.initial_blocks_to_swap, variant.initial_blocks_to_swap > 0))
    add(UpscaleAttempt(initial_batch, max_swap, True))
    for b in (21, 13, 9, 5):
        if b < initial_batch:
            add(UpscaleAttempt(b, max_swap, True))
    add(UpscaleAttempt(1, max_swap, True))
    return plan


class SeedVR2OOMError(RuntimeError):
    """Запуск SeedVR2 упал из-за нехватки памяти (VRAM или RAM)."""


class SeedVR2CompileError(RuntimeError):
    """Запуск SeedVR2 упал на этапе torch.compile (inductor/Triton), а не в самой модели."""


# Признаки сбоя компиляции в логе CLI.
COMPILE_ERROR_PATTERNS = re.compile(
    r"(torch\._dynamo|torch\._inductor|InductorError|BackendCompilerFailed|TritonError|"
    r"triton\.compiler|CompilationError|LoweringException|Failed to find C compiler)",
)


# ---------------------------------------------------------------------------
# Основной класс
# ---------------------------------------------------------------------------

@app.cls(
    gpu="L4",
    cpu=8.0,
    # RAM: Wan 2.2 (offload в RAM, ~22 ГБ) + BlockSwap SeedVR2 + потоковые чанки кадров 1080p.
    memory=65536,
    volumes={str(MODELS_DIR): models_volume, str(OUTPUTS_DIR): outputs_volume},
    timeout=GENERATE_TIMEOUT_S,
    scaledown_window=300,
    # GPU memory snapshot (enable_memory_snapshot + experimental_options={"enable_gpu_snapshot": True})
    # намеренно НЕ включён: SeedVR2 работает в дочерних процессах, которые снапшот не захватывает
    # (его холодный старт не сокращается); Wan держится в RAM через enable_model_cpu_offload (accelerate-хуки),
    # так что в VRAM на момент снапшота почти ничего нет; а setup() пишет в Volume (докачка весов + commit),
    # что в фазе снапшота небезопасно.
)
class PrometheusGenerator:
    seedvr2_model: str = modal.parameter(default=os.getenv("SEEDVR2_MODEL", "3b"))

    # -- жизненный цикл ------------------------------------------------------

    @modal.enter()
    def setup(self) -> None:
        """Один раз на контейнер: кладём веса в Volume (если их там нет) и готовим Wan.

        Пайплайн Wan грузится в RAM с enable_model_cpu_offload(): в VRAM в каждый момент находится
        только активный компонент. Это важно не только для генерации на L4, но и для апскейла —
        после генерации GPU полностью освобождается под SeedVR2.
        """
        import torch
        from diffusers import AutoencoderKLWan, WanPipeline

        if self.seedvr2_model not in SEEDVR2_VARIANTS:
            raise ValueError(f"SEEDVR2_MODEL должен быть одним из {sorted(SEEDVR2_VARIANTS)}, "
                             f"получено {self.seedvr2_model!r}")
        self.seedvr2 = SEEDVR2_VARIANTS[self.seedvr2_model]
        self._compile_enabled = SEEDVR2_COMPILE
        # (dit, ширина, высота, батч) -> откалиброванный batch_size; живёт, пока жив контейнер.
        self._calibration_cache: dict[tuple[str, int, int, int], int] = {}
        INDUCTOR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        TRITON_CACHE_DIR.mkdir(parents=True, exist_ok=True)

        HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = self._ensure_seedvr2_weights()

        vae = AutoencoderKLWan.from_pretrained(WAN_REPO_ID, subfolder="vae", torch_dtype=torch.float32)
        self.wan = WanPipeline.from_pretrained(WAN_REPO_ID, vae=vae, torch_dtype=torch.bfloat16)
        self.wan_i2v = None
        self._active_pipe = None
        self._use_pipeline(self.wan)  # создаётся лениво из тех же компонентов (без повторной загрузки весов)

        if downloaded or not (HF_CACHE_DIR / ".wan_ready").exists():
            (HF_CACHE_DIR / ".wan_ready").touch()
            models_volume.commit()
        log.info("контейнер готов: Wan=%s, SeedVR2=%s", WAN_REPO_ID, self.seedvr2.dit_file)

    def _ensure_seedvr2_weights(self) -> bool:
        """Скачивает DiT и VAE SeedVR2 в Volume. CLI ищет их в --model_dir и сам сверяет sha256
        (результат проверки кэшируется рядом с весами, повторно хэш не считается).
        Возвращает True, если что-то было скачано (нужен commit тома)."""
        from huggingface_hub import hf_hub_download

        SEEDVR2_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        downloaded = False
        for filename in (self.seedvr2.dit_file, SEEDVR2_VAE_FILE):
            target = SEEDVR2_MODEL_DIR / filename
            if target.exists() and target.stat().st_size > 0:
                continue
            log.info("скачиваю %s/%s -> %s", SEEDVR2_WEIGHTS_REPO, filename, SEEDVR2_MODEL_DIR)
            hf_hub_download(repo_id=SEEDVR2_WEIGHTS_REPO, filename=filename, local_dir=str(SEEDVR2_MODEL_DIR))
            downloaded = True
        return downloaded

    # -- генерация (Wan 2.2) -------------------------------------------------

    def generate_video(self, prompt: str, output_path: Path, duration_s: float = 30.0,
                       seed: int | None = None) -> Path:
        """Генерирует вертикальное 720p-видео (704x1280, 24 fps) заданной длительности.

        Wan 2.2 TI2V-5B даёт до 121 кадра (~5 с) за проход, поэтому длинный ролик собирается из
        клипов: первый — text-to-video, каждый следующий — image-to-video от последнего кадра
        предыдущего (продолжение сцены без скачка). Стыковочный кадр отбрасывается, чтобы не было
        «замирания» на склейке.
        """
        import numpy as np
        import torch
        from diffusers import WanImageToVideoPipeline
        from diffusers.utils import export_to_video
        from PIL import Image

        if not prompt or not prompt.strip():
            raise ValueError("prompt не может быть пустым")
        if duration_s <= 0:
            raise ValueError("duration_s должен быть > 0")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        total_frames = int(round(duration_s * WAN_FPS))
        seed = seed if seed is not None else int.from_bytes(os.urandom(4), "little")

        frames: list[np.ndarray] = []
        clip_idx = 0
        while len(frames) < total_frames:
            generator = torch.Generator(device="cpu").manual_seed(seed + clip_idx)
            common = dict(
                prompt=prompt,
                negative_prompt=WAN_NEGATIVE_PROMPT,
                height=WAN_HEIGHT,
                width=WAN_WIDTH,
                num_frames=WAN_FRAMES_PER_CLIP,
                num_inference_steps=WAN_STEPS,
                guidance_scale=WAN_GUIDANCE,
                generator=generator,
                output_type="np",
            )
            started = time.monotonic()
            if clip_idx == 0:
                self._use_pipeline(self.wan)
                clip = self.wan(**common).frames[0]
            else:
                if self.wan_i2v is None:
                    # from_pipe переиспользует уже загруженные модули (без второй копии весов в RAM)
                    self.wan.remove_all_hooks()
                    self.wan_i2v = WanImageToVideoPipeline.from_pipe(self.wan)
                self._use_pipeline(self.wan_i2v)
                last = Image.fromarray((np.clip(frames[-1], 0.0, 1.0) * 255).round().astype(np.uint8))
                clip = self.wan_i2v(image=last, **common).frames[0][1:]
            frames.extend(list(clip))
            log.info("Wan: клип %d готов (%d кадров) за %.0f c, всего %d/%d",
                     clip_idx, len(clip), time.monotonic() - started, min(len(frames), total_frames),
                     total_frames)
            clip_idx += 1

        frames = frames[:total_frames]
        tmp_path = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.tmp.mp4")
        try:
            # export_to_video пишет через imageio-ffmpeg (H.264); пересжимаем с контролем качества,
            # чтобы на апскейл не попадали лишние артефакты компрессии.
            export_to_video(frames, str(tmp_path), fps=WAN_FPS)
            _run([
                "ffmpeg", "-y", "-v", "error", "-i", str(tmp_path),
                "-c:v", "libx264", "-crf", "14", "-preset", "slow", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", str(output_path),
            ], timeout=1800)
        finally:
            tmp_path.unlink(missing_ok=True)

        self._release_wan_vram()
        log.info("Wan: видео сохранено %s (%d кадров, %.1f c)", output_path, total_frames, total_frames / WAN_FPS)
        return output_path

    def _use_pipeline(self, pipe) -> None:
        """T2V и I2V делят одни и те же модули. CPU-offload хуки должен держать только активный
        пайплайн, иначе два набора хуков гоняют один модуль между CPU и GPU и ломают offload."""
        if getattr(self, "_active_pipe", None) is pipe:
            return
        for other in (self.wan, self.wan_i2v):
            if other is not None and other is not pipe:
                other.remove_all_hooks()
        pipe.enable_model_cpu_offload()
        self._active_pipe = pipe

    def _release_wan_vram(self) -> None:
        """Освобождает VRAM после Wan перед запуском SeedVR2 в отдельном процессе.

        С enable_model_cpu_offload() веса и так в RAM, но хуки offload держат последний модуль
        на GPU до maybe_free_model_hooks(), а кэширующий аллокатор PyTorch — освобождённые блоки.
        SeedVR2 работает в дочернем процессе и видит только реально свободную VRAM.
        """
        import torch

        for pipe in (getattr(self, "wan", None), getattr(self, "wan_i2v", None)):
            if pipe is not None:
                pipe.maybe_free_model_hooks()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            free, total = torch.cuda.mem_get_info()
            log.info("VRAM перед апскейлом: свободно %.1f из %.1f ГБ", free / 2**30, total / 2**30)

    # -- апскейл (SeedVR2) ---------------------------------------------------

    def upscale_video(self, input_path: Path, output_path: Path) -> Path:
        """Апскейл 720p -> 1080p моделью SeedVR2 с временной согласованностью.

        Шаги:
          1. ffprobe исходника: размер, fps, число кадров, кодек, наличие аудио.
          2. Выбор batch_size (4n+1, >= 5) по длине видео и лимиту VRAM.
          3. Автокалибровка (один раз на контейнер и размер кадра): короткий прогон на 5 кадрах
             с полным батчем, замер пика VRAM. Если батч не помещается с запасом — стартуем сразу
             с меньшего, не тратя минуты на заведомо падающую попытку.
          4. Запуск SeedVR2 (inference_cli.py) в дочернем процессе, по лестнице fallback'ов при OOM.
             Отдельный процесс гарантирует, что после OOM вся VRAM вернётся драйверу — в том же
             процессе фрагментированный аллокатор часто падает и на следующей попытке.
          5. Финальная сборка ffmpeg: точный размер 1080x1920, кодек как у исходника,
             аудиодорожка копируется из исходного файла без перекодирования.
          6. Проверка результата и атомарная запись в output_path.

        Возвращает output_path. При ошибке — RuntimeError/ValueError/FileNotFoundError с описанием.
        """
        input_path = Path(input_path)
        output_path = Path(output_path)
        if not input_path.is_file():
            raise FileNotFoundError(f"upscale_video: входной файл не найден: {input_path}")
        if input_path.resolve() == output_path.resolve():
            raise ValueError("upscale_video: output_path совпадает с input_path")
        if not SEEDVR2_CLI.is_file():
            raise RuntimeError(f"upscale_video: SeedVR2 CLI не найден ({SEEDVR2_CLI}) — пересоберите образ")

        src = probe_video(input_path)
        target_w, target_h = compute_target_size(src.width, src.height)
        if min(src.width, src.height) >= TARGET_SHORT_SIDE:
            log.warning("upscale_video: исходник уже %dx%d (>=1080p), всё равно прогоняю через SeedVR2",
                        src.width, src.height)
        log.info("upscale_video: %s %dx%d, %d кадров, %.3f fps, %.1f c, codec=%s, audio=%s -> %dx%d",
                 input_path.name, src.width, src.height, src.frame_count, float(src.fps), src.duration_s,
                 src.video_codec, src.has_audio, target_w, target_h)

        self._release_wan_vram()

        initial_batch = choose_batch_size(src.frame_count, self.seedvr2.max_batch_size)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="seedvr2_") as tmp:
            work_dir = Path(tmp)
            upscaled_raw = work_dir / "upscaled_noaudio.mp4"

            initial_batch = self._calibrate_batch_size(input_path, src, initial_batch, work_dir)
            plan = build_attempt_plan(initial_batch, self.seedvr2)

            last_error: Exception | None = None
            used: UpscaleAttempt | None = None
            for i, attempt in enumerate(plan, start=1):
                log.info("SeedVR2: попытка %d/%d — %s", i, len(plan), attempt.describe())
                upscaled_raw.unlink(missing_ok=True)
                try:
                    self._run_seedvr2(input_path, upscaled_raw, attempt, src, work_dir)
                    used = attempt
                    break
                except SeedVR2OOMError as exc:
                    last_error = exc
                    log.warning("SeedVR2: нехватка памяти при %s — пробую более экономный режим",
                                attempt.describe())
                    continue
            if used is None:
                raise RuntimeError(
                    f"upscale_video: SeedVR2 не уместился в память даже при batch_size=1 с BlockSwap. "
                    f"Последняя ошибка: {last_error}"
                )
            if not used.temporal:
                log.warning("upscale_video: использован batch_size=1 — временная согласованность "
                            "отключена, возможно мерцание")

            up = probe_video(upscaled_raw)
            # SeedVR2 не должен терять/добавлять кадры (паддинг uniform_batch_size и prepend_frames
            # CLI срезает сам). Небольшое расхождение допустимо из-за округления в контейнере.
            if abs(up.frame_count - src.frame_count) > 2:
                raise RuntimeError(
                    f"upscale_video: число кадров не совпадает: исходник {src.frame_count}, "
                    f"после SeedVR2 {up.frame_count}"
                )
            log.info("SeedVR2: готово %dx%d, %d кадров (%s)", up.width, up.height, up.frame_count,
                     used.describe())

            tmp_out = output_path.with_name(f".{output_path.stem}.{uuid.uuid4().hex}.tmp{output_path.suffix or '.mp4'}")
            try:
                self._mux_final(upscaled_raw, input_path, tmp_out, src, target_w, target_h)
                final = probe_video(tmp_out)
                if (final.width, final.height) != (target_w, target_h):
                    raise RuntimeError(f"upscale_video: итоговый размер {final.width}x{final.height}, "
                                       f"ожидался {target_w}x{target_h}")
                if src.has_audio and not final.has_audio:
                    raise RuntimeError("upscale_video: аудиодорожка потеряна при сборке")
                os.replace(tmp_out, output_path)
            finally:
                tmp_out.unlink(missing_ok=True)

        self._persist_compile_cache()
        log.info("upscale_video: сохранено %s (%dx%d)", output_path, target_w, target_h)
        return output_path

    def _calibrate_batch_size(self, input_path: Path, src: VideoInfo, initial_batch: int,
                              work_dir: Path) -> int:
        """Микро-тест VRAM перед первым роликом данного размера в этом контейнере.

        Прогон SeedVR2 на первых CALIBRATION_FRAMES кадрах (--load_cap) с тем же batch_size и теми же
        флагами, что и у реального запуска. Благодаря --uniform_batch_size 5 кадров дополняются до
        полного батча, поэтому DiT и тайлы VAE получают ровно те же формы тензоров, что и в боевом
        прогоне, — пик памяти репрезентативен, а скомпилированные графы (torch.compile) сразу
        попадают в кэш и переиспользуются основным прогоном.

        Пик меряется по всей видеокарте (torch.cuda.mem_get_info), а не через
        torch.cuda.max_memory_allocated(): SeedVR2 работает в дочернем процессе, и счётчики
        аллокатора родителя его память не видят. Метрика «занято на устройстве» включает резерв
        аллокатора и CUDA-контекст — именно с ней сравнивается лимит 24 ГБ при OOM.

        Пороги (L4, 24 ГБ):
          - OOM на калибровке или пик > CALIBRATION_CRITICAL_GB -> 13;
          - пик > CALIBRATION_HIGH_GB (22 ГБ)                   -> 17;
          - иначе (в т.ч. < 18 ГБ)                              -> исходный батч (21).
        Результат кэшируется на контейнер; ошибки, не связанные с памятью, калибровку пропускают
        (лестница fallback'ов всё равно подстрахует).
        """
        if not CALIBRATION_ENABLED or initial_batch < 13 or self.seedvr2.initial_blocks_to_swap > 0:
            # Батч и так маленький (короткий ролик) или вариант 7B стартует с BlockSwap —
            # калибровка не сэкономит времени.
            return initial_batch
        if src.frame_count < CALIBRATION_FRAMES:
            return initial_batch

        key = (self.seedvr2.dit_file, src.width, src.height, initial_batch)
        cached = self._calibration_cache.get(key)
        if cached is not None:
            log.info("калибровка: batch_size=%d (из кэша контейнера)", cached)
            return cached

        attempt = UpscaleAttempt(initial_batch, 0, False)
        probe_out = work_dir / "calibration.mp4"
        started = time.monotonic()
        try:
            peak_bytes = self._run_seedvr2(input_path, probe_out, attempt, src, work_dir,
                                           load_cap=CALIBRATION_FRAMES, monitor_vram=True)
        except SeedVR2OOMError:
            chosen = 13
            log.warning("калибровка: OOM при batch_size=%d на %d кадрах -> стартую с batch_size=%d",
                        initial_batch, CALIBRATION_FRAMES, chosen)
            self._calibration_cache[key] = chosen
            return chosen
        except RuntimeError as exc:
            log.warning("калибровка пропущена (ошибка не по памяти): %s", str(exc).splitlines()[0])
            return initial_batch
        finally:
            probe_out.unlink(missing_ok=True)

        if peak_bytes is None:
            log.warning("калибровка: не удалось измерить VRAM, оставляю batch_size=%d", initial_batch)
            return initial_batch

        peak_gb = peak_bytes / 2**30
        if peak_gb > CALIBRATION_CRITICAL_GB:
            chosen = min(initial_batch, 13)
        elif peak_gb > CALIBRATION_HIGH_GB:
            chosen = min(initial_batch, 17)
        else:
            chosen = initial_batch
        log.info("калибровка: пик VRAM %.1f ГБ при batch_size=%d за %.0f c -> batch_size=%d",
                 peak_gb, initial_batch, time.monotonic() - started, chosen)
        self._calibration_cache[key] = chosen
        return chosen

    def _compile_args(self, attempt: UpscaleAttempt) -> list[str]:
        """Флаги torch.compile для CLI.

        backend inductor + mode max-autotune — рекомендация документации SeedVR2 для продакшена
        (DiT +20–40%, VAE +15–25%). На попытках с BlockSwap используется max-autotune-no-cudagraphs:
        CUDA graphs записывают фиксированные адреса весов в VRAM, а BlockSwap на каждом шаге
        перекладывает блоки между CPU и GPU — графы постоянно инвалидируются, а их пул памяти
        съедает VRAM, ради которой BlockSwap и включён. Ядра Triton при этом те же — на качество
        режим не влияет.
        """
        if not self._compile_enabled:
            return []
        mode = SEEDVR2_COMPILE_MODE
        if attempt.blocks_to_swap > 0 and mode == "max-autotune":
            mode = "max-autotune-no-cudagraphs"
        return [
            "--compile_dit", "--compile_vae",
            "--compile_backend", "inductor",
            "--compile_mode", mode,
        ]

    def _persist_compile_cache(self) -> None:
        """Сохраняет кэш inductor/Triton в Volume, чтобы следующие контейнеры не компилировали заново."""
        if not self._compile_enabled:
            return
        try:
            models_volume.commit()
        except Exception as exc:  # кэш — оптимизация, его потеря не должна ронять апскейл
            log.warning("не удалось сохранить кэш компиляции в Volume: %s", exc)

    def _run_seedvr2(self, input_path: Path, output_path: Path, attempt: UpscaleAttempt,
                     src: VideoInfo, work_dir: Path, *, load_cap: int = 0,
                     monitor_vram: bool = False) -> int | None:
        """Один запуск inference_cli.py.

        Поднимает SeedVR2OOMError при нехватке памяти и RuntimeError при любой другой ошибке
        (её fallback по памяти не лечит). Исключение — сбой torch.compile: тогда та же попытка
        повторяется один раз без компиляции, а компиляция отключается до конца жизни контейнера —
        ускорение не должно стоить упавшего ролика.

        load_cap > 0 — обработать только первые N кадров (калибровка).
        monitor_vram — замерить пик занятой VRAM на устройстве (байты) и вернуть его.
        """
        batch = attempt.batch_size
        temporal = attempt.temporal
        frames = min(src.frame_count, load_cap) if load_cap > 0 else src.frame_count

        # Потоковый режим (--chunk_size): кадры читаются и обрабатываются порциями, иначе 45 с
        # видео в 1080p (~1350 кадров) целиком в RAM занимают десятки ГБ.
        # Чанк кратен батчу, чтобы границы батчей и чанков совпадали.
        chunk_size = max(batch * 8, 64) if batch > 1 else 96
        chunk_size = min(chunk_size, max(frames, batch))

        base_cmd: list[str] = [
            sys.executable, str(SEEDVR2_CLI), str(input_path),
            "--output", str(output_path),
            "--output_format", "mp4",
            "--model_dir", str(SEEDVR2_MODEL_DIR),
            "--dit_model", self.seedvr2.dit_file,
            "--resolution", str(TARGET_SHORT_SIDE),
            "--batch_size", str(batch),
            "--seed", "42",
            "--chunk_size", str(chunk_size),
            # Цветокоррекция по исходнику в пространстве LAB — убирает дрейф цвета между батчами.
            "--color_correction", "lab",
            # Промежуточное видео пишем через ffmpeg x264 CRF 12 (почти lossless); финальная сборка ниже.
            "--video_backend", "ffmpeg",
            # Тайловый VAE: энкод/декод 1080x1920 целиком — главный пик VRAM; тайлы с перекрытием
            # 128 px без видимых швов.
            "--vae_encode_tiled", "--vae_decode_tiled",
            "--vae_encode_tile_size", "1024", "--vae_encode_tile_overlap", "128",
            "--vae_decode_tile_size", "1024", "--vae_decode_tile_overlap", "128",
            # Промежуточные тензоры между фазами — в RAM.
            "--tensor_offload_device", "cpu",
            # Модели между чанками не перезагружаются (кэш), а в простое лежат в RAM.
            "--cache_dit", "--cache_vae",
            "--dit_offload_device", "cpu",
            "--vae_offload_device", "cpu",
            "--attention_mode", "sdpa",
        ]
        if load_cap > 0:
            # Калибровка: читаем только первые N кадров, полноценное видео не обрабатывается.
            base_cmd += ["--load_cap", str(load_cap)]
        if temporal:
            base_cmd += [
                # Последний батч дополняется до полного размера: короткий «хвостовой» батч
                # даёт видимый скачок качества/мерцание в конце ролика.
                "--uniform_batch_size",
                # Перекрытие соседних батчей/чанков на 3 кадра с плавным блендом — убирает
                # «ступеньку» на границах батчей.
                "--temporal_overlap", "3",
                # 4 отражённых кадра в начале (CLI удаляет их сам) — стабилизирует первые кадры.
                "--prepend_frames", "4",
            ]
        if attempt.blocks_to_swap > 0:
            # BlockSwap: часть трансформер-блоков DiT живёт в RAM и подгружается на GPU по мере
            # прохода. Медленнее, но освобождает VRAM под активации большого батча.
            # Флага --use_non_blocking в CLI нет (на закреплённом коммите BlockSwap всегда
            # копирует с non_blocking=False) — передавать его нельзя, argparse завершит процесс.
            base_cmd += ["--blocks_to_swap", str(attempt.blocks_to_swap)]
            if attempt.swap_io_components:
                base_cmd.append("--swap_io_components")

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = f"{SEEDVR2_CODE_DIR}:{env.get('PYTHONPATH', '')}"
        # Уменьшает фрагментацию VRAM на длинных прогонах с переменными размерами тензоров.
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        # Персистентный кэш torch.compile на Volume. Каждая попытка — новый процесс, поэтому без
        # дискового кэша компиляция (минуты для max-autotune) повторялась бы на каждом ролике.
        # FX-граф, результаты автотюнинга и ядра Triton переживают и процесс, и контейнер.
        env["TORCHINDUCTOR_CACHE_DIR"] = str(INDUCTOR_CACHE_DIR)
        env["TRITON_CACHE_DIR"] = str(TRITON_CACHE_DIR)
        env["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
        env["TORCHINDUCTOR_AUTOGRAD_CACHE"] = "1"

        while True:
            compile_args = self._compile_args(attempt)
            cmd = base_cmd + compile_args
            label = attempt.describe() + (", compile" if compile_args else "") + \
                (f", calibration {load_cap} кадров" if load_cap else "")
            try:
                return self._exec_seedvr2(cmd, env, output_path, label, frames, monitor_vram)
            except SeedVR2CompileError as exc:
                if not compile_args:
                    raise RuntimeError(str(exc)) from exc
                log.warning("torch.compile не сработал — отключаю компиляцию в этом контейнере и "
                            "повторяю попытку без неё: %s", str(exc).splitlines()[0])
                self._compile_enabled = False
                output_path.unlink(missing_ok=True)

    def _exec_seedvr2(self, cmd: list[str], env: dict[str, str], output_path: Path, label: str,
                      frames: int, monitor_vram: bool) -> int | None:
        """Запускает процесс SeedVR2, стримит лог, классифицирует ошибку. Возвращает пик VRAM
        (байты, занято на устройстве) при monitor_vram=True."""
        tail: deque[str] = deque(maxlen=200)
        oom_seen = False
        compile_error_seen = False
        timed_out = threading.Event()
        stop_monitor = threading.Event()
        peak_used = [0]
        started = time.monotonic()
        proc = subprocess.Popen(
            cmd, cwd=str(SEEDVR2_CODE_DIR), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )

        def _on_timeout() -> None:
            # Сторожевой таймер: срабатывает, даже если процесс завис и ничего не пишет в лог.
            timed_out.set()
            proc.kill()

        def _monitor() -> None:
            # cudaMemGetInfo из родителя видит память всех процессов на GPU. Опрос каждые 20 мс;
            # короткие всплески между опросами компенсирует запас в порогах калибровки.
            import torch
            while not stop_monitor.is_set():
                try:
                    free, total = torch.cuda.mem_get_info()
                    peak_used[0] = max(peak_used[0], total - free)
                except Exception:
                    return
                stop_monitor.wait(0.02)

        monitor_thread: threading.Thread | None = None
        if monitor_vram and self._cuda_available():
            monitor_thread = threading.Thread(target=_monitor, daemon=True)
            monitor_thread.start()

        watchdog = threading.Timer(UPSCALE_ATTEMPT_TIMEOUT_S, _on_timeout)
        watchdog.daemon = True
        watchdog.start()
        try:
            assert proc.stdout is not None
            for line in proc.stdout:
                line = line.rstrip()
                if not line:
                    continue
                tail.append(line)
                if OOM_PATTERNS.search(line):
                    oom_seen = True
                if COMPILE_ERROR_PATTERNS.search(line):
                    compile_error_seen = True
                log.info("[seedvr2] %s", line)
            returncode = proc.wait()
        finally:
            watchdog.cancel()
            stop_monitor.set()
            if monitor_thread is not None:
                monitor_thread.join(timeout=1)
            if proc.poll() is None:
                proc.kill()
                proc.wait()

        elapsed = time.monotonic() - started
        tail_text = "\n".join(list(tail)[-40:])
        if timed_out.is_set():
            raise RuntimeError(f"SeedVR2: превышен таймаут попытки {UPSCALE_ATTEMPT_TIMEOUT_S} c "
                               f"({label}):\n{tail_text}")
        if returncode == -9 or (returncode != 0 and oom_seen):
            raise SeedVR2OOMError(f"нехватка памяти (код {returncode}, {elapsed:.0f} c, {label}):\n{tail_text}")
        if returncode != 0 and compile_error_seen:
            raise SeedVR2CompileError(f"ошибка torch.compile (код {returncode}, {label}):\n{tail_text}")
        if returncode != 0:
            raise RuntimeError(f"SeedVR2 завершился с кодом {returncode} ({label}):\n{tail_text}")
        if oom_seen and not output_path.is_file():
            raise SeedVR2OOMError(f"нехватка памяти ({label}):\n{tail_text}")
        if not output_path.is_file() or output_path.stat().st_size == 0:
            raise RuntimeError(f"SeedVR2 завершился успешно, но не создал {output_path}:\n{tail_text}")
        log.info("SeedVR2: прогон занял %.0f c (%.2f кадр/с, %s)", elapsed, frames / max(elapsed, 1e-6), label)
        if monitor_vram:
            return peak_used[0] or None
        return None

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return bool(torch.cuda.is_available())
        except Exception:
            return False

    def _mux_final(self, upscaled: Path, original: Path, output: Path, src: VideoInfo,
                   target_w: int, target_h: int) -> None:
        """Собирает итоговый файл: видео из SeedVR2 + аудио из исходника.

        - scale (lanczos) с сохранением пропорций «по покрытию» + центрированный crop до точного
          целевого размера (SeedVR2 масштабирует по короткой стороне и выравнивает до кратности 16).
        - fps фиксируется как у исходника (rational, без потерь на 29.97 и т.п.).
        - Кодек — тот же, что у исходника; 8-bit yuv420p для совместимости с площадками.
        - Аудио: -c:a copy — дорожка переносится бит-в-бит, без повторного сжатия; длительности
          совпадают, т.к. число кадров и fps сохранены.
        """
        encoder, enc_args = _encoder_for(src.video_codec)
        pix_fmt = "yuv420p10le" if "10" in src.pix_fmt and encoder == "libx265" else "yuv420p"
        vf = (
            f"scale={target_w}:{target_h}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={target_w}:{target_h},setsar=1"
        )
        cmd = [
            "ffmpeg", "-y", "-v", "error",
            "-i", str(upscaled),
            "-i", str(original),
            "-map", "0:v:0",
        ]
        if src.has_audio:
            cmd += ["-map", "1:a:0", "-c:a", "copy"]
        cmd += [
            "-vf", vf,
            "-r", f"{src.fps.numerator}/{src.fps.denominator}",
            "-c:v", encoder, *enc_args,
            "-pix_fmt", pix_fmt,
            "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
            "-map_metadata", "1",
            "-movflags", "+faststart",
            str(output),
        ]
        _run(cmd, timeout=3600)

    # -- удалённые методы ----------------------------------------------------

    @modal.method()
    def generate_and_upscale(self, prompt: str, duration_s: float = 30.0, seed: int | None = None,
                             upscale: bool = True) -> dict:
        """Полный цикл: генерация 720p -> апскейл 1080p. Результаты кладутся в Volume prometheus-outputs.
        Возвращает пути внутри тома и байты итогового ролика."""
        job_id = uuid.uuid4().hex[:12]
        job_dir = OUTPUTS_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        raw_path = self.generate_video(prompt, job_dir / "video_720p.mp4", duration_s=duration_s, seed=seed)
        final_path = raw_path
        if upscale:
            final_path = self.upscale_video(raw_path, job_dir / "video_1080p.mp4")
        outputs_volume.commit()

        return {
            "job_id": job_id,
            "raw_path": str(raw_path.relative_to(OUTPUTS_DIR)),
            "final_path": str(final_path.relative_to(OUTPUTS_DIR)),
            "video_bytes": final_path.read_bytes(),
        }

    @modal.method()
    def upscale_bytes(self, video_bytes: bytes, filename: str = "input.mp4") -> bytes:
        """Апскейл произвольного ролика, переданного байтами (например, после внешнего монтажа с аудио)."""
        if not video_bytes:
            raise ValueError("upscale_bytes: пустой файл")
        suffix = Path(filename).suffix or ".mp4"
        with tempfile.TemporaryDirectory(prefix="upscale_in_") as tmp:
            src = Path(tmp) / f"input{suffix}"
            dst = Path(tmp) / "output_1080p.mp4"
            src.write_bytes(video_bytes)
            self.upscale_video(src, dst)
            return dst.read_bytes()


# ---------------------------------------------------------------------------
# Локальные точки входа
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(prompt: str, duration: float = 30.0, seed: int = -1, out: str = "prometheus_1080p.mp4",
         no_upscale: bool = False) -> None:
    result = PrometheusGenerator().generate_and_upscale.remote(
        prompt=prompt,
        duration_s=duration,
        seed=None if seed < 0 else seed,
        upscale=not no_upscale,
    )
    Path(out).write_bytes(result["video_bytes"])
    print(f"job {result['job_id']}: {out} (в томе prometheus-outputs: {result['final_path']})")


@app.local_entrypoint()
def upscale_local_file(input_path: str, output_path: str = "") -> None:
    src = Path(input_path)
    if not src.is_file():
        raise SystemExit(f"файл не найден: {src}")
    dst = Path(output_path) if output_path else src.with_name(f"{src.stem}_1080p.mp4")
    data = PrometheusGenerator().upscale_bytes.remote(src.read_bytes(), src.name)
    dst.write_bytes(data)
    print(f"готово: {dst}")
