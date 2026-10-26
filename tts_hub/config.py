"""Service settings, read once from environment variables and CLI flags.

Service-wide variables are ``TTS_*``; each still falls back to its pre-rename ``BREEZE_*`` name
(``TTS_PORT`` → ``BREEZE_PORT``), the new name winning when both are set. Engine-specific
variables carry the engine's prefix: ``BREEZE_MODEL``, ``BREEZE_CODE_DIR``, ``QWEN_*_MODEL``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    names = (name, "BREEZE_" + name[4:]) if name.startswith("TTS_") else (name,)
    for n in names:
        value = os.environ.get(n)
        if value not in (None, ""):
            return value
    return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in ("1", "true", "yes", "on")


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 7861
    # Engine used when a request names none: breeze | qwen3
    default_engine: str = "breeze"
    # HuggingFace repo ids or local checkpoint directories
    model: str = "BreezeBlue/breeze-tts-2"
    qwen_custom_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
    qwen_base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    qwen_design_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
    code_dir: Path = PROJECT_ROOT / "vendor" / "breeze-tts"
    data_dir: Path = PROJECT_ROOT / "data"
    device: str = "auto"
    # Optional bearer token; empty = no auth (keep the service on 127.0.0.1 then)
    api_key: str = ""
    # Public base URL used in returned audio_url fields (default: derived from the request)
    public_url: str = ""
    # Upper bound on rows per generate() call. The actual size of each batch comes from the
    # fitted VRAM model against the VRAM free at that moment (batch_limits); lower this only to
    # leave VRAM for other programs.
    max_batch: int = 128
    # Long texts are split at sentence boundaries into chunks of at most this many chars.
    max_chunk_chars: int = 120
    # Seed the built-in voices (frozen design voices) after the model loads.
    seed_builtin_voices: bool = True
    # Load the model in a background thread at startup (False: on the first request).
    preload: bool = True
    # Idle lifecycle (seconds; 0 disables). With no API request for ``idle_unload_s`` the
    # weights leave the GPU (the next request reloads them, ~30 s incl. warmup); after a further
    # ``idle_shutdown_s`` without requests the process exits.
    idle_unload_s: float = 600
    idle_shutdown_s: float = 1800
    idle_check_s: float = 5
    extra: dict = field(default_factory=dict)

    @property
    def voices_dir(self) -> Path:
        return self.data_dir / "voices"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def jobs_dir(self) -> Path:
        return self.data_dir / "jobs"

    @property
    def vram_fit(self) -> Path:
        """Per-GPU VRAM model written by scripts/calibrate_vram.py (optional)."""
        return self.data_dir / "vram_fit.json"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            host=_env("TTS_HOST", cls.host),
            port=_env_int("TTS_PORT", cls.port),
            default_engine=_env("TTS_ENGINE", cls.default_engine).strip().lower(),
            model=_env("BREEZE_MODEL", cls.model),
            qwen_custom_model=_env("QWEN_CUSTOM_MODEL", cls.qwen_custom_model),
            qwen_base_model=_env("QWEN_BASE_MODEL", cls.qwen_base_model),
            qwen_design_model=_env("QWEN_DESIGN_MODEL", cls.qwen_design_model),
            code_dir=Path(_env("BREEZE_CODE_DIR", str(PROJECT_ROOT / "vendor" / "breeze-tts"))),
            data_dir=Path(_env("TTS_DATA_DIR", str(PROJECT_ROOT / "data"))),
            device=_env("TTS_DEVICE", cls.device),
            api_key=_env("TTS_API_KEY", ""),
            public_url=_env("TTS_PUBLIC_URL", "").rstrip("/"),
            max_batch=max(1, _env_int("TTS_MAX_BATCH", cls.max_batch)),
            max_chunk_chars=max(20, _env_int("TTS_MAX_CHUNK_CHARS", cls.max_chunk_chars)),
            seed_builtin_voices=_env_bool("TTS_SEED_VOICES", True),
            preload=_env_bool("TTS_PRELOAD", True),
            idle_unload_s=max(0.0, _env_float("TTS_IDLE_UNLOAD_MINUTES", cls.idle_unload_s / 60) * 60),
            idle_shutdown_s=max(0.0, _env_float("TTS_IDLE_SHUTDOWN_MINUTES", cls.idle_shutdown_s / 60) * 60),
        )

    def ensure_dirs(self) -> None:
        for path in (self.voices_dir, self.outputs_dir, self.jobs_dir):
            path.mkdir(parents=True, exist_ok=True)
