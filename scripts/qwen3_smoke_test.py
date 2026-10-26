"""End-to-end Qwen3-TTS checks against a running service (real GPU).

    .venv\\Scripts\\python.exe scripts\\qwen3_smoke_test.py [--url http://127.0.0.1:7861]
"""
from __future__ import annotations

import argparse
import re
import sys
import time

import httpx


def rate(text: str, seconds: float) -> str:
    han = len(re.findall(r"[一-鿿]", text))
    units = f"{han / seconds:.1f} 字/秒" if han else f"{len(text.split()) / seconds:.1f} 词/秒"
    return units


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7861")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.url, timeout=900)
    failures = 0

    def step(name, fn):
        nonlocal failures
        t = time.time()
        try:
            detail = fn()
            print(f"[OK]   {name:<30} {time.time() - t:6.1f}s  {detail}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name:<30} {time.time() - t:6.1f}s  {exc}")

    def ok(r):
        if r.status_code >= 300:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    def tts(**body):
        r = ok(c.post("/v1/tts", json=body))["result"]
        if r["engine"] != "qwen3":
            raise RuntimeError(f"engine={r['engine']}")
        return f"{r['duration_s']}s ({rate(body['text'], r['duration_s'])}) warn={r['warnings']}"

    presets = [v["id"] for v in ok(c.get("/v1/voices"))["voices"] if v["kind"] == "preset"]
    step("preset voices listed", lambda: f"{len(presets)}: {', '.join(presets)}")
    text = "今天的天气真不错，我们一起去公园散散步吧。"
    step("preset (cold load CustomVoice)", lambda: tts(text=text, voice="qwen_vivian", format="mp3"))
    step("preset warm", lambda: tts(text=text, voice="qwen_serena"))
    step("preset + instruct", lambda: tts(text="你怎么能这样对我！", voice="qwen_uncle_fu",
                                          instruction="非常愤怒，大声质问"))
    step("preset dialect (dylan)", lambda: tts(text="嘿，哥们儿，今儿个咱上哪儿溜达去？", voice="qwen_dylan"))
    step("preset english (ryan)", lambda: tts(text="Welcome back. Tonight we have a very special story for you.",
                                             voice="qwen_ryan", language="en"))
    step("clone on qwen3 (cold Base)", lambda: tts(text=text, voice="zh_male_narrator", engine="qwen3"))
    step("clone + instruct warning", lambda: tts(text=text, voice="zh_male_narrator", engine="qwen3",
                                                  instruction="开心"))

    def design():
        d = ok(c.post("/v1/voices/design", json={
            "voice_id": "smoke_witch", "name": "女巫", "engine": "qwen3", "overwrite": True,
            "instruction": "一位神秘的老年女巫，声音沙哑尖细，语速缓慢，带着诡异的笑意。"}))
        return f"{d['id']} kind={d['kind']} engine={d.get('design_engine')} ref={d['ref_seconds']}s"
    step("design voice via VoiceDesign", design)

    def dub():
        lines = [
            {"voice": "zh_male_narrator", "text": "夜色渐深，森林里起了雾。"},
            {"voice": "smoke_witch", "speaker": "女巫", "text": "迷路的孩子，过来吧。", "engine": "qwen3"},
            {"voice": "qwen_vivian", "speaker": "小雨", "text": "我、我才不怕你！", "instruction": "害怕但嘴硬"},
            {"voice": "zh_young_male", "speaker": "阿明", "text": "快跑，别回头！"},
        ]
        r = ok(c.post("/v1/dub", json={"lines": lines, "name": "mixed", "format": "mp3"}))["result"]
        spans = ", ".join(f"{t['engine']}:{t['duration_s']}s" for t in r["lines"])
        return f"{r['duration_s']}s [{spans}] {r['path']} warn={r['warnings']}"
    step("dub mixing breeze + qwen3", dub)

    def batch():
        lines = [{"voice": ["qwen_vivian", "qwen_serena", "qwen_uncle_fu", "qwen_eric"][i % 4],
                  "text": "这是批量合成的第几句台词，用来测试千问的批处理速度。"} for i in range(12)]
        t = time.time()
        r = ok(c.post("/v1/dub", json={"lines": lines, "save_lines": False}))["result"]
        el = time.time() - t
        return f"12 rows {el:.1f}s → {r['duration_s']}s audio ({r['duration_s'] / el:.1f}x realtime) warn={r['warnings']}"
    step("qwen3 batch 12 preset rows", batch)

    def openai():
        r = c.post("/v1/audio/speech", json={"model": "qwen3-tts", "input": "兼容接口测试。",
                                             "voice": "qwen_sohee", "response_format": "wav"})
        if r.status_code != 200:
            raise RuntimeError(r.text[:200])
        return f"{len(r.content)} bytes"
    step("openai model=qwen3-tts", openai)

    c.delete("/v1/voices/smoke_witch")
    h = c.get("/health").json()
    print("engines:", h["engines"], "vram:", h["vram"])
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
