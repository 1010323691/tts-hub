"""Voice library persisted under ``data/voices/<id>/`` (``voice.json`` + ``reference.wav``).

Kinds:
  - ``clone``  reference audio + exact transcript. Every line sounds like the reference; an
               optional default ``instruction`` turns synthesis into voice direction (Breeze
               only — Qwen3's Base model ignores instructions).
  - ``design`` a natural-language description only. Each synthesis re-imagines the voice, so
               timbre can drift between lines; create it with ``freeze`` (the default) to
               record one sample as the reference and get a consistent clone voice instead.
  - ``preset`` a Qwen3-TTS CustomVoice speaker (virtual, not stored on disk; ``qwen_<name>``).

Clone and design voices work on every engine; presets only on ``qwen3``.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from pathlib import Path

import numpy as np

from . import audio
from .qwen_engine import PRESETS as QWEN_PRESETS

VOICE_ID = re.compile(r"^[\w\-]{1,64}$")
REF_FILE = "reference.wav"
MIN_REF_SECONDS = 1.0
MAX_REF_SECONDS = 60.0

# Built-in voices: frozen design voices seeded after the model loads (see JobManager).
BUILTIN_VOICES = [
    {"id": "zh_female_narrator", "name": "中文女声旁白", "language": "zh",
     "instruction": "一位温柔知性的年轻女性，音色清亮柔和，吐字清晰，语速适中，适合有声书旁白。",
     "sample": "大家好，欢迎收听今天的故事。故事开始于一个安静的小镇，那里的人们过着平凡而温暖的生活。"},
    {"id": "zh_male_narrator", "name": "中文男声旁白", "language": "zh",
     "instruction": "一位沉稳醇厚的中年男性，嗓音低沉有磁性，语速平稳，适合纪录片旁白。",
     "sample": "在这片辽阔的土地上，四季更替，万物生长。每一条河流，都在讲述着属于自己的故事。"},
    {"id": "zh_young_male", "name": "青年男声", "language": "zh",
     "instruction": "一位阳光开朗的青年男性，声音清爽有活力，语气自然亲切。",
     "sample": "嘿，好久不见！最近过得怎么样？我刚从外地回来，给你带了点特产。"},
    {"id": "zh_young_female", "name": "青年女声", "language": "zh",
     "instruction": "一位活泼自信的年轻女性，声音明亮甜美，语气轻快而富有感染力。",
     "sample": "今天天气真好，我们一起去公园散散步吧！听说那边的花都开了，特别漂亮。"},
    {"id": "zh_girl", "name": "小女孩", "language": "zh",
     "instruction": "一个活泼可爱的七八岁小女孩，声音清脆稚嫩，语气天真俏皮。",
     "sample": "妈妈妈妈，你看！天上那朵云像不像一只大白兔？我也想飞上去摸一摸它。"},
    {"id": "zh_old_man", "name": "老爷爷", "language": "zh",
     "instruction": "一位慈祥的老爷爷，声音略带沙哑，语速缓慢，语气温和而有阅历。",
     "sample": "孩子啊，人这一辈子，走过的路多了，就会明白，平平安安才是最大的福气。"},
    {"id": "en_female_narrator", "name": "English female narrator", "language": "en",
     "instruction": "A warm, clear young woman with a calm, friendly and engaging narration style.",
     "sample": "Welcome back. Tonight's story begins in a quiet village by the sea, "
               "where every morning started with the sound of waves."},
    {"id": "en_male_narrator", "name": "English male narrator", "language": "en",
     "instruction": "A deep, steady middle-aged man with a confident, measured documentary narration style.",
     "sample": "For thousands of years, these mountains have watched over the valley below, "
               "silent witnesses to the passing of time."},
]


PRESET_VOICES = {
    f"qwen_{p['speaker']}": {
        "id": f"qwen_{p['speaker']}", "name": p["name"], "kind": "preset", "speaker": p["speaker"],
        "description": p["description"], "language": p["language"], "builtin": True,
        "consistent": True, "created_at": 0, "updated_at": 0,
    }
    for p in QWEN_PRESETS
}
ENGINES = ("breeze", "qwen3")


def voice_engines(voice: dict) -> list[str]:
    return ["qwen3"] if voice.get("kind") == "preset" else list(ENGINES)


class VoiceError(ValueError):
    pass


def slugify(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "_", (name or "").strip()).strip("_")[:48]
    return slug or f"voice_{int(time.time())}"


class VoiceStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    # -- read -------------------------------------------------------------------------------
    def _dir(self, voice_id: str) -> Path:
        if not VOICE_ID.match(voice_id or ""):
            raise VoiceError(f"非法音色 id：{voice_id!r}（仅允许字母、数字、下划线、连字符，≤64）")
        return self.root / voice_id

    def get(self, voice_id: str) -> dict | None:
        if voice_id in PRESET_VOICES:
            return dict(PRESET_VOICES[voice_id])
        try:
            path = self._dir(voice_id) / "voice.json"
        except VoiceError:
            return None
        if not path.is_file():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def require(self, voice_id: str) -> dict:
        voice = self.get(voice_id)
        if voice is None:
            raise VoiceError(f"音色不存在：{voice_id}（可用：{', '.join(v['id'] for v in self.list())}）")
        return voice

    def list(self) -> list[dict]:
        voices = []
        for child in sorted(self.root.iterdir()) if self.root.is_dir() else []:
            if child.is_dir():
                voice = self.get(child.name)
                if voice:
                    voices.append(voice)
        voices.sort(key=lambda v: (not v.get("builtin"), v.get("created_at", 0)))
        return voices + [dict(v) for v in PRESET_VOICES.values()]

    def reference_path(self, voice: dict) -> Path:
        return self._dir(voice["id"]) / REF_FILE

    def load_reference(self, voice: dict) -> tuple[np.ndarray, int]:
        return audio.read_audio(self.reference_path(voice))

    # -- write ------------------------------------------------------------------------------
    def create_clone(self, voice_id: str, name: str, ref_audio: bytes | Path, ref_text: str, *,
                     instruction: str = "", description: str = "", language: str = "",
                     overwrite: bool = False, extra: dict | None = None) -> dict:
        ref_text = (ref_text or "").strip()
        if not ref_text:
            raise VoiceError("克隆音色必须提供 ref_text（参考音频的逐字文本）")
        try:
            wav, sr = audio.read_audio(ref_audio)
        except ValueError as exc:
            raise VoiceError(f"参考音频无法解码：{exc}") from exc
        duration = wav.size / sr
        if not MIN_REF_SECONDS <= duration <= MAX_REF_SECONDS:
            raise VoiceError(f"参考音频时长 {duration:.1f}s，需在 {MIN_REF_SECONDS:.0f}–{MAX_REF_SECONDS:.0f}s 之间"
                             "（推荐 5–15s 清晰人声）")
        record = {
            "id": voice_id, "name": name or voice_id, "kind": "clone",
            "ref_text": ref_text, "instruction": (instruction or "").strip(),
            "description": description, "language": language,
            "ref_seconds": round(duration, 2), "consistent": True,
            **(extra or {}),
        }
        return self._save(record, wav, sr, overwrite)

    def create_design(self, voice_id: str, name: str, instruction: str, *, description: str = "",
                      language: str = "", overwrite: bool = False) -> dict:
        instruction = (instruction or "").strip()
        if not instruction:
            raise VoiceError("设计音色必须提供 instruction（音色描述）")
        record = {"id": voice_id, "name": name or voice_id, "kind": "design",
                  "instruction": instruction, "description": description, "language": language,
                  "consistent": False}
        return self._save(record, None, None, overwrite)

    def _save(self, record: dict, wav, sr, overwrite: bool) -> dict:
        if record["id"] in PRESET_VOICES:
            raise VoiceError(f"{record['id']} 是 Qwen3 预置音色的保留 id，请换一个")
        directory = self._dir(record["id"])
        with self._lock:
            if directory.exists():
                existing = self.get(record["id"])
                if not overwrite:
                    raise VoiceError(f"音色已存在：{record['id']}（传 overwrite=true 覆盖）")
                if existing and existing.get("builtin") and not record.get("builtin"):
                    raise VoiceError(f"内置音色不可覆盖：{record['id']}")
                shutil.rmtree(directory)
            directory.mkdir(parents=True)
            now = time.time()
            record.setdefault("builtin", False)
            record["created_at"] = now
            record["updated_at"] = now
            if wav is not None:
                (directory / REF_FILE).write_bytes(audio.encode(wav, "wav", sr))
            (directory / "voice.json").write_text(json.dumps(record, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
        return record

    def delete(self, voice_id: str) -> None:
        voice = self.require(voice_id)
        if voice.get("builtin"):
            raise VoiceError(f"内置音色不可删除：{voice_id}")
        with self._lock:
            shutil.rmtree(self._dir(voice_id), ignore_errors=True)


def public_view(voice: dict) -> dict:
    """The fields an API caller sees (no internal paths)."""
    keys = ("id", "name", "kind", "builtin", "consistent", "language", "description", "speaker",
            "instruction", "ref_text", "ref_seconds", "source", "design_instruction", "design_engine",
            "created_at")
    view = {k: voice[k] for k in keys if k in voice and voice[k] not in (None, "")}
    view["engines"] = voice_engines(voice)
    return view
