"""VRAM-fitted batch sizing shared by both engines (pure, stdlib-only).

Peak VRAM of one tensor batch is modelled as

    base + base_per_frame × frames + rows × multiplier × (row_fixed_bytes + bytes_per_row_frame × frames)
    frames = prompt_frames + text_chars + decode_cap(text_chars)      (the batch's longest row)

``budget`` is what a batch may use: free + reserved − allocated, read right before planning (so
another loaded engine is accounted for). A batch is sized to ``headroom`` of the predicted
limit; the planner fills each batch shortest-first until the next row would exceed it.

Qwen3-TTS constants are NarrifyAudio's fit (``backend/core/tts_batch_limits.py``, 2026-10-12,
RTX 5090, qwen-tts 0.1.1 with the talker hidden-state trim, ``expandable_segments:True``): every
row was forced to decode to its cap, residuals within 0.4 %. Breeze constants come from
``scripts/calibrate_vram.py`` on the same GPU; ``data/vram_fit.json`` (written by that script)
overrides them for another GPU.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path

MAX_ROWS = 128               # both engines' largest supported batch
CPU_FALLBACK_ROWS = 4        # no CUDA budget to plan with
# Narrify: a clone voice of unknown length (and preset/design prompts) is planned as a 10 s
# reference at 4.73 chars/s: 125 + 47 prompt positions.
PRODUCTION_PROMPT_FRAMES = 12.5 * 10.0 + round(10.0 * 4.73)


@dataclass(frozen=True)
class VramModel:
    name: str
    row_fixed_bytes: float
    bytes_per_row_frame: float
    base_bytes: float = 0.0       # per-batch transient independent of rows (graphs, workspaces)
    base_bytes_per_frame: float = 0.0  # per-batch transient growing with length, not rows
    multiplier: int = 1           # Breeze CFG runs a cond + uncond row per text row
    headroom: float = 0.9

    def row_bytes(self, frames: float) -> float:
        return self.multiplier * (self.row_fixed_bytes + self.bytes_per_row_frame * float(frames))

    def batch_bytes(self, frames: float) -> float:
        return self.base_bytes + self.base_bytes_per_frame * float(frames)

    def predicted_bytes(self, rows: int, frames: float) -> float:
        return self.batch_bytes(frames) + rows * self.row_bytes(frames)

    def rows_limit(self, frames: float, budget: float) -> int:
        """Rows that exactly fill ``budget`` (no headroom)."""
        return max(0, int((float(budget) - self.batch_bytes(frames)) // self.row_bytes(frames)))

    def cap(self, frames: float, budget: float, max_rows: int = MAX_ROWS) -> int:
        """Production rows: ``headroom`` of the limit, rounded down to even, at least 1."""
        rows = int(self.headroom * self.rows_limit(frames, budget))
        rows -= rows % 2
        return max(1, min(int(max_rows), rows))


# -- Qwen3-TTS (NarrifyAudio fit) -------------------------------------------------------------
QWEN3 = VramModel("qwen3", row_fixed_bytes=3_042_372, bytes_per_row_frame=123_971)
# Without the hidden-state trim the talker keeps all 29 layers' states (2048 wide, bf16).
QWEN3_UNTRIMMED = replace(QWEN3, name="qwen3-untrimmed",
                          bytes_per_row_frame=QWEN3.bytes_per_row_frame + 29 * 2048 * 2)

# -- Breeze TTS 2 (scripts/calibrate_vram.py, RTX 5090, 2026-10-26) ----------------------------
# 30 batches (1–96 rows, 12/48/110 chars, plain clone and CFG design). Rows ≥ 24 fit the per-row
# terms within ±5 %; the per-frame batch term is the envelope of small batches (one 110-char row
# alone peaks at ~1 GiB), so no sample is under-predicted. Per-row cost (~110 KiB/frame) is close
# to Qwen3's: both decode through the same 12.5 Hz codec.
BREEZE_PLAIN = VramModel("breeze", row_fixed_bytes=3_699_925, bytes_per_row_frame=111_628,
                         base_bytes=77_225_308, base_bytes_per_frame=1_780_862)
BREEZE_CFG = VramModel("breeze-cfg", row_fixed_bytes=0, bytes_per_row_frame=114_135,
                       base_bytes=98_229_452, base_bytes_per_frame=1_928_872, multiplier=2)


def load_overrides(path: Path) -> dict[str, VramModel]:
    """Models from ``data/vram_fit.json`` ({name: {row_fixed_bytes, bytes_per_row_frame, ...}})."""
    out: dict[str, VramModel] = {}
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out
    for name, fields in data.get("models", {}).items():
        try:
            out[name] = VramModel(name=name, **{k: fields[k] for k in fields if k in VramModel.__dataclass_fields__ and k != "name"})
        except TypeError:
            continue
    return out


def plan(positions: list[int], prompt_frames, chars, decode_cap, model: VramModel | None,
         budget: float | None, max_rows: int = MAX_ROWS) -> list[list[int]]:
    """Cut ``positions`` (already one generate-compatible group) into VRAM-safe batches.

    Rows go shortest-first; a batch grows while the model, evaluated at the batch's longest
    prompt and longest text, still admits one more row. Without a budget (CPU) batches hold
    ``CPU_FALLBACK_ROWS``.
    """
    ordered = sorted(positions, key=lambda p: (chars(p), prompt_frames(p)))
    batches: list[list[int]] = []
    current: list[int] = []
    longest_prompt, longest_chars = 0.0, 0
    for p in ordered:
        np_, nc = max(longest_prompt, prompt_frames(p)), max(longest_chars, chars(p))
        if model is None or not budget:
            limit = min(max_rows, CPU_FALLBACK_ROWS)
        else:
            limit = model.cap(np_ + nc + decode_cap(nc), budget, max_rows)
        if current and len(current) + 1 > limit:
            batches.append(current)
            current, np_, nc = [], prompt_frames(p), chars(p)
        current.append(p)
        longest_prompt, longest_chars = np_, nc
    if current:
        batches.append(current)
    return batches


def oom_retry_rows(rows: int) -> int:
    """Rows re-run at once after an OOM: all but the longest 10 % (at least one pushed back)."""
    return max(1, rows - max(1, -(-rows // 10)))


def vram_budget(torch, device) -> float | None:
    """free + reserved − allocated on ``device`` (bytes), ``None`` off CUDA."""
    if device is None or not str(device).startswith("cuda") or not torch.cuda.is_available():
        return None
    free, _total = torch.cuda.mem_get_info(device)
    return float(free + torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device))
