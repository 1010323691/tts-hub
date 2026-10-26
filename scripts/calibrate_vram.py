"""Fit Breeze TTS 2's batch VRAM model on this GPU and write ``data/vram_fit.json``.

Stop the service first (it holds VRAM), then:

    .venv\\Scripts\\python.exe scripts\\calibrate_vram.py [--quick]

For every mode (plain clone; CFG = direction / design) and a grid of rows x text lengths it runs
one real batch, records the peak allocated bytes above the loaded-model baseline, and fits

    peak = base + rows x multiplier x (row_fixed + per_frame x frames)
    frames = prompt frames + instruction chars + text chars + decoded frames (longest row)

Rows ≥ 24 fit the per-row terms by least squares (the regime that decides batch size); small
batches also show a per-batch transient that grows with length but not with rows (~1 GiB for one
110-char row), so ``base_bytes_per_frame`` is the envelope that keeps every sample at or below the
prediction. Planning evaluates the formula at the decode *cap* (worst case), so natural-length
measurements give a conservative model. Each run starts without a captured depth-decoder graph,
so the graph's own memory is part of the measured peak.

``--refit`` recomputes the models from the samples already in the output file (no GPU needed).
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tts_hub import audio  # noqa: E402
from tts_hub.config import Settings  # noqa: E402
from tts_hub.engine import BreezeEngine, Row  # noqa: E402
from tts_hub.text import text_chars  # noqa: E402

STORY = ("夜色渐深，古堡里传来一阵奇怪的声响，大家都屏住了呼吸。小雨紧紧抓住阿明的衣袖，低声说她听见走廊尽头有脚步声。"
         "阿明举起手电筒，光柱在石墙上晃动，照出一幅落满灰尘的旧画像。画中的伯爵似乎正盯着他们，嘴角带着若有若无的冷笑。"
         "两人对视一眼，谁也没有说话，只听见窗外的风越刮越紧，吹得木门吱呀作响。")


def text_of(n: int) -> str:
    s = (STORY * 4)[:max(1, n - 1)]
    return s + "。"


def fit(samples: list[dict]) -> dict:
    """Per-mode VramModel fields from measured samples (see module docstring)."""
    models = {}
    for mode, mult, name in (("plain", 1, "breeze"), ("cfg", 2, "breeze-cfg")):
        pts = [x for x in samples if x["mode"] == mode]
        big = [x for x in pts if x["rows"] >= 24]
        if len(big) < 3:
            continue
        a = np.array([[1.0, x["rows"] * mult, x["rows"] * mult * x["frames"]] for x in big])
        y = np.array([x["peak"] for x in big], dtype=float)
        base, fixed, per_frame = (float(c) for c in np.linalg.lstsq(a, y, rcond=None)[0])
        base, fixed, per_frame = max(0.0, base), max(0.0, fixed), max(1.0, per_frame)

        def linear(x):
            return base + x["rows"] * mult * (fixed + per_frame * x["frames"])
        per_frame_base = max(0.0, max((x["peak"] - linear(x)) / x["frames"] for x in pts))
        worst = max((x["peak"] - linear(x) - per_frame_base * x["frames"]) / x["peak"] for x in pts)
        over = max((linear(x) + per_frame_base * x["frames"] - x["peak"]) / x["peak"] for x in big)
        print(f"{mode}: base={base / 2**20:.0f} MiB  base/frame={per_frame_base / 1024:.0f} KiB  "
              f"row={fixed / 2**20:.2f} MiB + {per_frame / 1024:.1f} KiB/frame  "
              f"worst under {max(0.0, worst):.1%}, worst over (rows≥24) {over:.1%}")
        models[name] = {"row_fixed_bytes": fixed, "bytes_per_row_frame": per_frame,
                        "base_bytes": base, "base_bytes_per_frame": per_frame_base,
                        "multiplier": mult, "headroom": 0.9}
    return models


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grid")
    ap.add_argument("--refit", action="store_true", help="refit from the samples in --out")
    ap.add_argument("--out", default=str(ROOT / "data" / "vram_fit.json"))
    args = ap.parse_args()
    out = Path(args.out)
    if args.refit:
        data = json.loads(out.read_text(encoding="utf-8"))
        data["models"] = fit(data["samples"])
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print("written", out)
        return 0
    import torch

    s = Settings.from_env()
    eng = BreezeEngine(s.model, s.code_dir, s.device, 128)
    eng.load()
    dev = eng.device
    ref = s.voices_dir / "zh_male_narrator" / "reference.wav"
    meta = json.loads((ref.parent / "voice.json").read_text(encoding="utf-8"))
    wav, sr = audio.read_audio(ref)
    prompt = eng.encode_reference(wav, sr, meta["ref_text"])
    prompt_frames = int(prompt["codes"].shape[0]) + text_chars(meta["ref_text"])
    instruction = "一位沉稳的中年男性，语速平稳，语气紧张而克制。"

    modes = {
        "plain": dict(prompt=prompt, instruction="", cfg=False, prompt_frames=prompt_frames),
        "cfg": dict(prompt=None, instruction=instruction, cfg=True, prompt_frames=text_chars(instruction)),
    }
    rows_grid = [1, 8, 24, 48] if args.quick else [1, 8, 24, 48, 96]
    chars_grid = [12, 48, 110]
    samples = []
    for mode, m in modes.items():
        for chars in chars_grid:
            for rows in rows_grid:
                batch = [Row(text_of(chars), m["prompt"], m["instruction"], None, 42) for _ in range(rows)]
                eng._depth.graph, eng._depth.bucket_size = None, None
                gc.collect()
                torch.cuda.synchronize(dev)
                torch.cuda.empty_cache()
                base = torch.cuda.memory_allocated(dev)
                torch.cuda.reset_peak_memory_stats(dev)
                t = time.time()
                try:
                    res = eng._generate(batch)
                except torch.OutOfMemoryError:
                    print(f"{mode:5s} chars={chars:3d} rows={rows:3d}  OOM — skipping longer grids", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                peak = torch.cuda.max_memory_allocated(dev) - base
                gen = max(r.frames for r in res)
                frames = m["prompt_frames"] + text_chars(batch[0].text) + gen
                samples.append({"mode": mode, "rows": rows, "chars": chars, "frames": frames,
                                "gen_frames": gen, "peak": peak, "seconds": round(time.time() - t, 1)})
                print(f"{mode:5s} chars={chars:3d} rows={rows:3d} frames={frames:4d} "
                      f"peak={peak / 2**30:6.2f} GiB  {time.time() - t:5.1f}s", flush=True)
                del res

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"gpu": torch.cuda.get_device_name(dev), "created": time.strftime("%Y-%m-%d %H:%M"),
                               "models": fit(samples), "samples": samples}, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print("written", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
