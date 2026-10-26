"""Qwen3-TTS engine (ported from NarrifyAudio ``tts-engine/tts_worker.py``), Apache-2.0 weights.

Qwen3-TTS splits the three modes across three 1.7B checkpoints, each loaded on first use:

  - preset  CustomVoice: built-in speakers, optional instruct (emotion / style)
  - clone   Base: reference audio + transcript (ICL). Base ignores instructs.
  - design  VoiceDesign: a natural-language voice description as the instruct

Same interface as ``BreezeEngine`` so the job queue, voice library and idle lifecycle are
shared. Rows of one mode batch through the list APIs, shortest-first, as many per call as
NarrifyAudio's fitted VRAM model admits (``batch_limits.QWEN3``; it assumes the talker
hidden-state trim installed below). In a tensor batch the talker does not stop rows at EOS, so
the decode cap follows the longest row's length (4.5 frames/char, NarrifyAudio's production
calibration on 19k rows); a row that decodes to its own cap is retried alone with another seed
at the 6 frames/char fallback.
"""
from __future__ import annotations

import gc
import logging
import os
from dataclasses import fields, is_dataclass, replace

import numpy as np

from . import batch_limits
from .engine import MAX_BATCH, Row, RowResult, group_rows, run_batches
from .text import CODEC_FRAME_RATE

log = logging.getLogger("tts_hub.qwen3")

SAMPLE_RATE = 24000
MAX_NEW_TOKENS = 2048
FRAME_CAP_FLOOR = 128
FRAMES_PER_CHAR = 4.5
FALLBACK_FRAMES_PER_CHAR = 6.0
RUNAWAY_RETRIES = 1
DEFAULT_DESIGN_INSTRUCTION = "音色清晰自然，语速平稳。"

# CustomVoice speakers (NarrifyAudio ``backend/core/voice_presets.py``); id = speaker name
PRESETS = [
    {"speaker": "vivian", "name": "Vivian", "description": "明亮、略带个性的年轻女声", "language": "zh"},
    {"speaker": "serena", "name": "Serena", "description": "温暖、柔和的年轻女声", "language": "zh"},
    {"speaker": "uncle_fu", "name": "Uncle Fu", "description": "低沉、醇厚的成熟男声", "language": "zh"},
    {"speaker": "dylan", "name": "Dylan", "description": "清晰、自然的年轻男声（北京话）", "language": "zh"},
    {"speaker": "eric", "name": "Eric", "description": "活泼、明亮且略带沙哑的男声（四川话）", "language": "zh"},
    {"speaker": "ryan", "name": "Ryan", "description": "富有动感和节奏感的男声", "language": "en"},
    {"speaker": "aiden", "name": "Aiden", "description": "阳光、清晰的美国男声", "language": "en"},
    {"speaker": "ono_anna", "name": "Ono Anna", "description": "俏皮、轻盈的日本女声", "language": "ja"},
    {"speaker": "sohee", "name": "Sohee", "description": "温暖、情感丰富的韩国女声", "language": "ko"},
]


def frame_cap(chars: int, per_char: float = FRAMES_PER_CHAR) -> int:
    return min(MAX_NEW_TOKENS, max(FRAME_CAP_FLOOR, int(chars * per_char)))


def qwen_batch_key(row: Row) -> tuple:
    # list APIs take per-row speaker / instruct / prompt / language, so only mode and seed split
    return (row.mode, row.seed)


TRIM_HIDDEN_QWEN_TTS_VERSION = "0.1.1"
_TRIM = {"installed": False, "active": False}


def install_talker_hidden_trim() -> bool:
    """Wrap the talker forward so decoding keeps only the hidden state it reads (idempotent).

    NarrifyAudio port. qwen-tts 0.1.1 hard-codes ``output_hidden_states=True`` for
    ``talker.generate``, so every decode step keeps all 29 layers' states until generation ends —
    about as much memory per frame as the KV cache. The caller only reads ``hid[0][-1][:, -1:]``
    (exactly ``past_hidden``) and ``hid[-1]`` (codec ids); the wrapper returns
    ``((past_hidden,), codec_ids)``: numerically identical, without the per-layer tuple. Only the
    audited qwen-tts version is wrapped. ``QWEN_TTS_TRIM_HIDDEN=0`` disables it. Returns whether
    the trim is active (the VRAM model to plan with depends on it).
    """
    if _TRIM["installed"]:
        return _TRIM["active"]
    _TRIM["installed"] = True
    if os.environ.get("QWEN_TTS_TRIM_HIDDEN", "1").strip().lower() in ("0", "false", "off", "no"):
        return False
    try:
        from importlib.metadata import version
        installed = version("qwen-tts")
    except Exception:  # noqa: BLE001 — metadata missing: do not touch an unknown library
        installed = ""
    if installed != TRIM_HIDDEN_QWEN_TTS_VERSION:
        log.warning("qwen-tts %s 未经审计（需要 %s），不裁剪 talker 隐藏状态，按未裁剪显存模型分批",
                    installed or "未知", TRIM_HIDDEN_QWEN_TTS_VERSION)
        return False
    from qwen_tts.core.models import modeling_qwen3_tts as modeling
    cls = modeling.Qwen3TTSTalkerForConditionalGeneration
    original = cls.forward

    def forward(self, *args, **kwargs):
        kwargs["output_hidden_states"] = False
        out = original(self, *args, **kwargs)
        out.hidden_states = ((out.past_hidden,), out.hidden_states[1])
        return out

    forward.__wrapped__ = original
    cls.forward = forward
    _TRIM["active"] = True
    log.info("已启用 talker 隐藏状态裁剪（qwen-tts 0.1.1，数值等价）")
    return True


def qwen_prompt_frames(row: Row) -> float:
    """Prompt positions of a row: a clone reference's codes + transcript, else Narrify's default."""
    prompt = row.prompt
    code = getattr(prompt, "ref_code", None)
    if code is not None:
        return float(int(code.shape[0]) + len(getattr(prompt, "ref_text", "") or ""))
    return float(batch_limits.PRODUCTION_PROMPT_FRAMES)


def _move(value, device):
    """Copy prompt tensors to ``device`` (dataclass / container aware)."""
    if is_dataclass(value) and not isinstance(value, type):
        return replace(value, **{f.name: _move(getattr(value, f.name), device) for f in fields(value)})
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_move(v, device) for v in value)
    if hasattr(value, "to") and hasattr(value, "detach"):
        return value.detach().to(device)
    return value


class Qwen3Engine:
    name = "qwen3"
    modes = ("preset", "clone", "design")

    def __init__(self, custom_model: str, base_model: str, design_model: str,
                 device: str = "auto", max_batch: int = MAX_BATCH, vram_fit=None):
        self.model_ids = {"preset": custom_model, "clone": base_model, "design": design_model}
        self.device_pref = device
        self.max_batch = max(1, min(int(max_batch), MAX_BATCH))
        overrides = batch_limits.load_overrides(vram_fit) if vram_fit else {}
        self._vram_override = overrides.get("qwen3")
        self.oom_scale = 1.0
        self.last_plan: list[int] = []
        self.models: dict = {}
        self.loaded = False
        self.device = None
        self._cuda = None

    # -- lifecycle --------------------------------------------------------------------------
    def load(self) -> None:
        """Cheap: resolve the device; each checkpoint loads on first use (``_model``)."""
        import torch

        device = self.device_pref
        if device in ("auto", "", None):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            device = "cuda:0"
        self.device = device
        self._cuda = device if device.startswith("cuda") else None
        self._torch = torch
        self.oom_scale = 1.0
        self.loaded = True

    @property
    def vram_model(self) -> batch_limits.VramModel:
        if self._vram_override is not None:
            return self._vram_override
        return batch_limits.QWEN3 if _TRIM["active"] else batch_limits.QWEN3_UNTRIMMED

    def unload(self) -> None:
        if not self.loaded:
            return
        self.models.clear()
        self.loaded = False
        gc.collect()
        if self._cuda is not None:
            self._torch.cuda.empty_cache()
        log.info("已卸载 Qwen3-TTS 模型")

    def loaded_models(self) -> list[str]:
        return sorted(self.models)

    def vram(self) -> dict | None:
        if self._cuda is None:
            return None
        torch = self._torch
        free, total = torch.cuda.mem_get_info(self._cuda)
        return {"allocated_gb": round(torch.cuda.memory_allocated(self._cuda) / 2**30, 2),
                "reserved_gb": round(torch.cuda.memory_reserved(self._cuda) / 2**30, 2),
                "free_gb": round(free / 2**30, 2), "total_gb": round(total / 2**30, 2)}

    def _model(self, mode: str):
        model = self.models.get(mode)
        if model is not None:
            return model
        from .engine import resolve_checkpoint

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from qwen_tts import Qwen3TTSModel

        torch = self._torch
        path = resolve_checkpoint(self.model_ids[mode])
        log.info("加载 Qwen3-TTS %s 模型 %s → %s", mode, path, self.device)
        kwargs = {"dtype": torch.bfloat16 if self._cuda else torch.float32}
        if self.device != "cpu":
            kwargs["device_map"] = self.device
        model = Qwen3TTSModel.from_pretrained(path, **kwargs)
        install_talker_hidden_trim()
        self.models[mode] = model
        return model

    # -- prompts ----------------------------------------------------------------------------
    def encode_reference(self, wav: np.ndarray, sr: int, ref_text: str):
        items = self._model("clone").create_voice_clone_prompt(
            ref_audio=(np.asarray(wav, dtype=np.float32).reshape(-1), int(sr)),
            ref_text=(ref_text or "").strip())
        return _move(items[0], "cpu")

    # -- synthesis --------------------------------------------------------------------------
    def synthesize(self, rows: list[Row], progress=None) -> list[RowResult]:
        batches = self.plan(rows)
        self.last_plan = [len(b) for b in batches]
        return run_batches(batches, rows, self._generate_checked, progress, self._on_oom)

    def plan(self, rows: list[Row]) -> list[list[int]]:
        """VRAM-sized batches per (mode, seed) group, against the VRAM free now."""
        groups = group_rows(rows, qwen_batch_key)
        for mode in {key[0] for key in groups}:
            self._model(mode)  # load first: the budget must not count VRAM a model is about to take
        budget = batch_limits.vram_budget(self._torch, self._cuda)
        if budget is not None:
            budget *= self.oom_scale
        batches = []
        for positions in groups.values():
            batches += batch_limits.plan(
                positions, lambda p: qwen_prompt_frames(rows[p]), lambda p: len(rows[p].text),
                frame_cap, self.vram_model, budget, self.max_batch)
        return batches

    def _on_oom(self) -> None:
        gc.collect()
        self._torch.cuda.empty_cache()
        self.oom_scale *= 0.9

    def _generate_checked(self, rows: list[Row]) -> list[RowResult]:
        results = self._generate(rows, FRAMES_PER_CHAR)
        for i, (row, res) in enumerate(zip(rows, results)):
            attempt = 0
            while res.capped and attempt < RUNAWAY_RETRIES:
                attempt += 1
                log.warning("第 %d 行解码到上限（%d 帧，%d 字），换种子单独重试", i, res.frames, len(row.text))
                retry = replace(row, seed=row.seed + 7919 * attempt)
                res = self._generate([retry], FALLBACK_FRAMES_PER_CHAR)[0]
            results[i] = res
        return results

    def _generate(self, rows: list[Row], per_char: float) -> list[RowResult]:
        torch = self._torch
        mode = rows[0].mode
        model = self._model(mode)
        texts = [r.text for r in rows]
        languages = [r.language or "auto" for r in rows]
        caps = [frame_cap(len(t), per_char) for t in texts]
        common = {"text": texts, "language": languages, "non_streaming_mode": True,
                  "max_new_tokens": max(caps)}
        torch.manual_seed(rows[0].seed)
        if mode == "preset":
            wavs, sr = model.generate_custom_voice(
                speaker=[r.speaker for r in rows], instruct=[r.instruction.strip() for r in rows], **common)
        elif mode == "clone":
            device = getattr(model, "device", None) or self.device
            wavs, sr = model.generate_voice_clone(
                voice_clone_prompt=[_move(r.prompt, device) for r in rows], **common)
        else:
            wavs, sr = model.generate_voice_design(
                instruct=[r.instruction.strip() or DEFAULT_DESIGN_INSTRUCTION for r in rows], **common)
        if len(wavs) != len(rows):
            raise RuntimeError(f"Qwen3-TTS 返回 {len(wavs)} 段音频，期望 {len(rows)} 段")
        out = []
        for wav, cap in zip(wavs, caps):
            wav = np.asarray(wav, dtype=np.float32).reshape(-1)
            if sr != SAMPLE_RATE:
                import librosa
                wav = librosa.resample(wav, orig_sr=sr, target_sr=SAMPLE_RATE)
            frames = int(round(wav.size * CODEC_FRAME_RATE / SAMPLE_RATE))
            out.append(RowResult(wav, frames, frames >= cap - 1))
        return out
