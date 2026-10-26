"""Breeze TTS 2 inference engine (ported from NarrifyAudio ``tts-engine/breeze_engine.py``).

One checkpoint serves every mode, chosen per row by what the row carries:

  - clone      reference codes + ref_text            (template ref_clone_tata, CFG 1)
  - direction  reference codes + ref_text + instruction (ref_edit_tata, CFG 4)
  - design     instruction only                      (tts_instruction, CFG 4)

Rows sharing a template, CFG and seed are batched in eager ``generate`` calls, shortest-first so
similar lengths share a batch. How many rows a call takes comes from the fitted VRAM model
(``batch_limits``) evaluated against the VRAM free right now; an OOM still splits the batch
(``run_batches``). Each batch's decode cap follows its longest row; a row that decodes to its
own cap (a runaway) is retried alone with another seed.

Not thread-safe: the job worker is the only caller. Importing this module is cheap; torch and
the vendored Breeze code load in ``BreezeEngine.load``.
"""
from __future__ import annotations

import gc
import logging
import os
import sys
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import batch_limits
from .text import CODEC_FRAME_RATE, frame_cap, text_chars

log = logging.getLogger("tts_hub.breeze")

SAMPLE_RATE = 24000
MAX_BATCH = 128             # BatchDepthDecoder's largest batch
INSTRUCTION_CFG = 4.0       # upstream recommends 4 whenever an instruction is given
CLONE_CFG = 1.0             # plain cloning runs unguided (its template has no negative branch)
REPETITION_PENALTY = 1.1
DEFAULT_DESIGN_INSTRUCTION = "音色清晰自然，语速平稳。"
RUNAWAY_RETRIES = 1
FAST_CLONE = os.environ.get("BREEZE_FAST_CLONE", "1") != "0"
_PROMPT_SCHEME = "breeze-prompt://"
# Reference codes for the generate call in progress, keyed by placeholder path. Module-level on
# purpose: the upstream hook must not reference an engine instance.
_PROMPT_CODES: dict = {}


@dataclass
class Row:
    """One utterance to synthesize (engine-agnostic; each engine reads what it supports)."""
    text: str
    prompt: object | None = None    # engine-specific clone prompt (Breeze: {"codes", "ref_text"})
    instruction: str = ""
    cfg_scale: float | None = None  # Breeze only. None: 4 with an instruction, else 1
    seed: int = 42
    speaker: str = ""               # Qwen3 CustomVoice preset speaker
    language: str = ""              # Qwen3 language name ("" = auto)

    @property
    def cfg(self) -> float:
        if self.cfg_scale is not None:
            return float(self.cfg_scale)
        return INSTRUCTION_CFG if self.instruction.strip() else CLONE_CFG

    @property
    def mode(self) -> str:
        if self.speaker:
            return "preset"
        return "clone" if self.prompt is not None else "design"


@dataclass
class RowResult:
    audio: np.ndarray
    frames: int
    capped: bool


def _install_prompt_codes_hook(templates) -> None:
    """Let the upstream prompt builder take pre-encoded reference codes (idempotent)."""
    if getattr(templates, "_tts_hub_prompt_codes", False):
        return
    original = templates._encode_prompt_audio

    def encode(audio_tokenizer, audio_path):
        codes = _PROMPT_CODES.get(audio_path)
        return codes if codes is not None else original(audio_tokenizer, audio_path)

    templates._encode_prompt_audio = encode
    templates._tts_hub_prompt_codes = True


def _cached_snapshot(model_id: str) -> str | None:
    """A complete snapshot folder of ``model_id`` in the HF cache, even without ``refs/main``."""
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:
        return None
    snapshots = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}" / "snapshots"
    if not snapshots.is_dir():
        return None
    for snap in sorted(snapshots.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if (snap / "config.json").is_file() and (snap / "audio_tokenizer" / "config.json").is_file():
            return str(snap)
    return None


def resolve_checkpoint(model_id: str) -> str:
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import snapshot_download
    try:
        return snapshot_download(model_id, local_files_only=True)
    except Exception:  # noqa: BLE001 — no resolvable cached revision
        pass
    cached = _cached_snapshot(model_id)
    if cached:
        return cached
    log.info("权重未缓存，正在从 HuggingFace 下载 %s（Breeze 约 7.7 GB，Qwen3 每个约 4.5 GB）…", model_id)
    try:
        return snapshot_download(model_id)
    except OSError as exc:
        if getattr(exc, "winerror", None) == 1314:
            raise RuntimeError("HuggingFace 缓存需要创建符号链接：请开启 Windows「开发者模式」或以管理员运行，"
                               "或先下载权重到本地目录并用 BREEZE_MODEL 指向它") from exc
        raise


def is_recoverable_oom(exc: BaseException) -> bool:
    """PyTorch's allocator OOM leaves the CUDA context usable; driver-level errors do not."""
    return type(exc).__name__ == "OutOfMemoryError"


def release_exception_frames(exc: BaseException) -> None:
    """Free the tensors a failed generate's traceback pins (its KV cache and activations)."""
    if exc.__traceback__ is not None:
        traceback.clear_frames(exc.__traceback__)
    exc.__traceback__ = exc.__context__ = exc.__cause__ = None


def run_batches(batches: list[list[int]], rows: list[Row], generate, progress=None, on_oom=None):
    """Run planned ``batches`` (positions into ``rows``) through ``generate``; results in row order.

    A recoverable OOM (NarrifyAudio's in-process split): the batch's shortest 90 % re-run at
    once and its longest rows follow as their own batch; ``on_oom`` lets the engine shrink its
    budget for later planning. A single row that still OOMs re-raises.
    """
    results: list = [None] * len(rows)
    pending = deque(batches)
    done = 0
    while pending:
        batch = pending.popleft()
        try:
            out = generate([rows[p] for p in batch])
        except Exception as exc:
            if not is_recoverable_oom(exc) or len(batch) == 1:
                raise
            release_exception_frames(exc)
            if on_oom is not None:
                on_oom()
            keep = batch_limits.oom_retry_rows(len(batch))
            log.warning("批次 %d 行显存不足：先重跑较短的 %d 行，最长的 %d 行另起一批",
                        len(batch), keep, len(batch) - keep)
            pending.appendleft(batch[keep:])
            pending.appendleft(batch[:keep])
            continue
        for pos, res in zip(batch, out):
            results[pos] = res
        done += len(batch)
        if progress:
            progress(done, len(rows))
    return results


def breeze_batch_key(row: Row) -> tuple:
    return (row.prompt is not None, bool(row.instruction.strip()), row.cfg, row.seed)


def breeze_prompt_frames(row: Row) -> float:
    """Prompt positions a row adds before its text: reference codes + transcript + instruction."""
    frames = text_chars(row.instruction)
    if isinstance(row.prompt, dict) and row.prompt.get("codes") is not None:
        frames += int(row.prompt["codes"].shape[0]) + text_chars(row.prompt.get("ref_text", ""))
    return float(frames)


def plan_batches(rows: list[Row], max_batch: int, key=breeze_batch_key) -> list[list[int]]:
    """Group row positions into generate calls (pure).

    Breeze needs one template, one CFG scale and one seed per call (a batch samples from a
    single RNG stream, so a row with its own seed gets its own call); ``key`` lets another
    engine group by its own constraints. Within a group rows go shortest-first and are cut
    into ``max_batch``-sized calls.
    """
    size = max(1, min(int(max_batch), MAX_BATCH))
    batches = []
    for positions in group_rows(rows, key).values():
        positions.sort(key=lambda p: text_chars(rows[p].text))
        batches.extend(positions[i:i + size] for i in range(0, len(positions), size))
    return batches


def group_rows(rows: list[Row], key) -> dict[tuple, list[int]]:
    groups: dict[tuple, list[int]] = {}
    for pos, row in enumerate(rows):
        groups.setdefault(key(row), []).append(pos)
    return groups


class BreezeEngine:
    name = "breeze"
    modes = ("clone", "design")  # clone rows also follow an instruction (voice direction)

    def __init__(self, model_id: str, code_dir: Path, device: str = "auto", max_batch: int = MAX_BATCH,
                 vram_fit: Path | None = None):
        self.model_id = model_id
        self.code_dir = Path(code_dir)
        self.device_pref = device
        self.max_batch = max(1, min(int(max_batch), MAX_BATCH))
        overrides = batch_limits.load_overrides(vram_fit) if vram_fit else {}
        self.vram_models = {False: overrides.get("breeze", batch_limits.BREEZE_PLAIN),
                            True: overrides.get("breeze-cfg", batch_limits.BREEZE_CFG)}
        self.oom_scale = 1.0  # shrinks after an OOM; reset by load()
        self.last_plan: list[int] = []
        self.loaded = False
        self.device = None
        self._cuda = None

    # -- lifecycle --------------------------------------------------------------------------
    def load(self) -> None:
        import torch

        source = str(self.code_dir)
        if not os.path.isfile(os.path.join(source, "breeze_infer", "runtime.py")):
            raise RuntimeError(f"未找到 Breeze 推理代码：{source}（请运行 install.ps1，"
                               "或用 BREEZE_CODE_DIR 指定目录）")
        if source not in sys.path:
            sys.path.insert(0, source)
        from breeze_infer import api as upstream_api
        from breeze_infer import templates
        from breeze_infer.batch import BatchDepthDecoder, eager_batch_modules
        from breeze_infer.runtime import (load_runtime, set_all_seeds,
                                          update_generation_config_for_breeze)
        from models.logits_process import GeneratedTokenRepetitionPenaltyLogitsProcessor
        from transformers.generation.logits_process import LogitsProcessorList

        device = self.device_pref
        if device in ("auto", "", None):
            device = "cuda:0" if torch.cuda.is_available() else "cpu"
        elif device == "cuda":
            device = "cuda:0"
        checkpoint = Path(resolve_checkpoint(self.model_id))
        log.info("加载 Breeze 模型 %s → %s", checkpoint, device)
        self.tokenizer, self.breeze, self.audio_tokenizer = load_runtime(
            checkpoint, device=device, attn_implementation="eager")
        update_generation_config_for_breeze(self.breeze)
        self.device = self.breeze.device
        # Capturing the depth-decoder graph casts both output heads to float32 in place; keep
        # checkpoint-dtype copies on CPU and swap them in per call, as the upstream batch API does.
        self._heads = upstream_api._checkpoint_head_weights(self.breeze)
        self._swapped = upstream_api._swapped_weights
        self._depth = BatchDepthDecoder(self.breeze, max_batch_size=MAX_BATCH)
        self._eager = eager_batch_modules
        self._templates = templates
        self._seed = set_all_seeds
        self._logits = lambda: LogitsProcessorList(
            [GeneratedTokenRepetitionPenaltyLogitsProcessor(REPETITION_PENALTY)])
        self._torch = torch
        self._cuda = self.device if str(self.device).startswith("cuda") else None
        _install_prompt_codes_hook(templates)
        self.oom_scale = 1.0
        self.loaded = True

    def unload(self) -> None:
        """Drop every model object and return the VRAM to the driver.

        The process keeps its CUDA context (a few hundred MB); ``load`` restores everything.
        """
        if not self.loaded:
            return
        for attr in ("tokenizer", "breeze", "audio_tokenizer", "_heads", "_depth"):
            setattr(self, attr, None)
        self.loaded = False
        self.device = None
        gc.collect()
        if self._cuda is not None:
            self._torch.cuda.empty_cache()
        log.info("已卸载 Breeze 模型权重")

    def vram(self) -> dict | None:
        cuda = getattr(self, "_cuda", None)
        if cuda is None:
            return None
        torch = self._torch
        free, total = torch.cuda.mem_get_info(cuda)
        return {"allocated_gb": round(torch.cuda.memory_allocated(cuda) / 2**30, 2),
                "reserved_gb": round(torch.cuda.memory_reserved(cuda) / 2**30, 2),
                "free_gb": round(free / 2**30, 2), "total_gb": round(total / 2**30, 2)}

    # -- prompts ----------------------------------------------------------------------------
    def encode_reference(self, wav: np.ndarray, sr: int, ref_text: str) -> dict:
        """Encode reference audio once; the result is reused for every row of that voice."""
        wav = np.asarray(wav, dtype=np.float32).reshape(-1)
        encoded = self.audio_tokenizer.encode(wav, sr=sr)
        codes = self._torch.as_tensor(encoded["audio_codes"][0], dtype=self._torch.int16)
        if codes.ndim != 2:
            raise ValueError(f"参考音频编码形状异常：{tuple(codes.shape)}")
        return {"codes": codes.cpu().contiguous(), "ref_text": (ref_text or "").strip()}

    # -- synthesis --------------------------------------------------------------------------
    def synthesize(self, rows: list[Row], progress=None) -> list[RowResult]:
        """Synthesize every row; results come back in input order."""
        batches = self.plan(rows)
        self.last_plan = [len(b) for b in batches]
        return run_batches(batches, rows, self._generate_checked, progress, self._on_oom)

    def plan(self, rows: list[Row]) -> list[list[int]]:
        """VRAM-sized batches per (template, CFG, seed) group, against the VRAM free now."""
        budget = batch_limits.vram_budget(self._torch, self._cuda)
        if budget is not None:
            budget *= self.oom_scale
        batches = []
        for positions in group_rows(rows, breeze_batch_key).values():
            model = self.vram_models[rows[positions[0]].cfg != 1.0]
            batches += batch_limits.plan(
                positions, lambda p: breeze_prompt_frames(rows[p]), lambda p: text_chars(rows[p].text),
                frame_cap, model, budget, self.max_batch)
        return batches

    def _on_oom(self) -> None:
        gc.collect()
        self._torch.cuda.empty_cache()
        self.oom_scale *= 0.9

    def _generate_checked(self, rows: list[Row]) -> list[RowResult]:
        results = self._generate(rows)
        for i, (row, res) in enumerate(zip(rows, results)):
            attempt = 0
            while res.capped and attempt < RUNAWAY_RETRIES:
                attempt += 1
                log.warning("第 %d 行解码到上限（%d 帧，%d 字），换种子单独重试", i, res.frames,
                            text_chars(row.text))
                retry = Row(row.text, row.prompt, row.instruction, row.cfg_scale, row.seed + 7919 * attempt)
                res = self._generate([retry])[0]
            results[i] = res
        return results

    @contextmanager
    def _graph_depth_without_cfg(self, cfg: float):
        """Route unguided depth decoding through the captured CFG graph.

        Upstream decodes each unguided frame with a full HF ``depth_decoder.generate`` call
        (~2.5x slower than real time on an RTX 5090). Feeding the graph cond == uncond at
        scale 1 gives ``u + 1 * (c - u) == c`` exactly, with the same reserved-token masking
        and sampling, so plain cloning gets the CFG path's speed. ``BREEZE_FAST_CLONE=0``
        restores the upstream path.
        """
        if cfg != 1.0 or not FAST_CLONE:
            yield
            return
        depth, graph = self.breeze.depth_decoder, self._depth

        def generate(input_ids=None, backbone_last_hidden_state=None, **_):
            return graph.generate_with_cfg(input_ids, backbone_last_hidden_state,
                                           backbone_last_hidden_state, 1.0)

        depth.generate = generate
        try:
            yield
        finally:
            del depth.generate

    def _generate(self, rows: list[Row]) -> list[RowResult]:
        torch, templates = self._torch, self._templates
        cfg = rows[0].cfg
        batch, codes = [], {}
        for n, r in enumerate(rows):
            req = {"id": f"row-{n}", "speaker": "S0", "text": r.text}
            instruction = r.instruction.strip()
            if r.prompt is None and not instruction:
                instruction = DEFAULT_DESIGN_INSTRUCTION
            if instruction:
                req["instruction"] = instruction
            if r.prompt is not None:
                key = f"{_PROMPT_SCHEME}{n}"
                codes[key] = r.prompt["codes"].cpu()
                req["ref_audio_path"], req["ref_text"] = key, r.prompt["ref_text"]
            batch.append(req)
        names = {templates.select_template_name(r) for r in batch}
        if len(names) != 1:
            raise ValueError(f"同一批提示模板不一致：{sorted(names)}")
        caps = [frame_cap(text_chars(r.text)) for r in rows]
        seed = rows[0].seed
        self._seed(seed)
        _PROMPT_CODES.update(codes)
        try:
            inputs = templates.prepare_inputs(
                self.tokenizer, self.audio_tokenizer, self.breeze, batch,
                templates.get_template(names.pop()), guidance_scale=cfg,
                guidance_scale_ref=None, guidance_scale_ins=None)
        finally:
            _PROMPT_CODES.clear()
        with self._eager(self.breeze):
            self._depth.prepare(len(batch))
            self._seed(seed)  # graph capture consumes RNG; reseed so results stay reproducible
            with torch.inference_mode(), self._swapped(self._heads), self._depth.bound(self.breeze), \
                    self._graph_depth_without_cfg(cfg):
                audio = self.breeze.generate(
                    **inputs, output_audio=True, audio_tokenizer=self.audio_tokenizer,
                    logits_processor=self._logits(), max_new_tokens=max(caps))
        if len(audio) != len(batch):
            raise RuntimeError(f"Breeze 返回 {len(audio)} 段音频，期望 {len(batch)} 段")
        out = []
        for a, cap in zip(audio, caps):
            wav = a.detach().float().cpu().numpy().reshape(-1)
            frames = int(round(wav.size * CODEC_FRAME_RATE / SAMPLE_RATE))
            out.append(RowResult(wav, frames, frames >= cap - 1))
        return out
