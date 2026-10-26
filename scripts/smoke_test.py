"""End-to-end smoke test against a running service (real GPU).

    .venv\\Scripts\\python.exe scripts\\smoke_test.py [--url http://127.0.0.1:7861] [--api-key KEY]
"""
from __future__ import annotations

import argparse
import sys
import time

import httpx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7861")
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()
    headers = {"Authorization": f"Bearer {args.api_key}"} if args.api_key else {}
    c = httpx.Client(base_url=args.url, headers=headers, timeout=900)
    failures = 0

    def step(name, fn):
        nonlocal failures
        t = time.time()
        try:
            detail = fn()
            print(f"[OK]   {name:<28} {time.time() - t:6.1f}s  {detail}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[FAIL] {name:<28} {time.time() - t:6.1f}s  {exc}")

    def ok(r):
        if r.status_code >= 300:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()

    step("health", lambda: ok(c.get("/health"))["status"])
    step("voices", lambda: len(ok(c.get("/v1/voices"))["voices"]))

    def tts_single():
        d = ok(c.post("/v1/tts", json={"text": "[笑] 欢迎来到今晚的故事时间，让我们一起开始吧。",
                                       "voice": "zh_young_female", "format": "mp3"}))
        r = d["result"]
        return f"{r['duration_s']}s {r['path']} warn={r['warnings']}"
    step("tts single (mp3)", tts_single)

    def tts_long():
        text = "从前有一座山，山里有一座庙，庙里有个老和尚在给小和尚讲故事。讲的是什么呢？" * 6
        r = ok(c.post("/v1/tts", json={"text": text, "voice": "zh_old_man"}))["result"]
        return f"{r['duration_s']}s chunks={len(r['chunks'])} warn={r['warnings']}"
    step("tts long (chunked)", tts_long)

    def tts_direction():
        r = ok(c.post("/v1/tts", json={"text": "你怎么能这样对我！我再也不相信你了！",
                                       "voice": "zh_young_female", "instruction": "愤怒地大声质问，语速很快"}))["result"]
        return f"{r['duration_s']}s"
    step("tts voice direction", tts_direction)

    def tts_en():
        r = ok(c.post("/v1/tts", json={"text": "(sigh) It is good to hear your voice again after all this time.",
                                       "voice": "en_male_narrator"}))["result"]
        return f"{r['duration_s']}s"
    step("tts english", tts_en)

    def design():
        d = ok(c.post("/v1/voices/design", json={"voice_id": "smoke_villain", "name": "反派",
                                                 "instruction": "一位阴险狡诈的中年男性反派，声音低沉沙哑，语速缓慢，带着冷笑。",
                                                 "overwrite": True}))
        return f"{d['id']} kind={d['kind']} ref={d['ref_seconds']}s"
    step("design voice (frozen)", design)

    def clone():
        ref = c.get("/v1/voices/zh_male_narrator/audio").content
        v = ok(c.get("/v1/voices/zh_male_narrator"))
        r = c.post("/v1/voices", data={"voice_id": "smoke_clone", "name": "克隆测试", "ref_text": v["ref_text"],
                                       "overwrite": "true"},
                   files={"ref_audio": ("ref.wav", ref, "audio/wav")})
        d = ok(r)
        return f"{d['id']} ref={d['ref_seconds']}s"
    step("clone voice (upload)", clone)

    def dub():
        lines = [
            {"voice": "zh_male_narrator", "text": "夜色渐深，古堡里传来一阵奇怪的声响。"},
            {"voice": "zh_young_female", "speaker": "小雨", "text": "你听到了吗？好像有人在走廊里。",
             "instruction": "紧张害怕，压低声音小声说"},
            {"voice": "zh_young_male", "speaker": "阿明", "text": "别怕，有我在。我们一起去看看。"},
            {"voice": "smoke_villain", "speaker": "伯爵", "text": "[笑] 欢迎来到我的城堡，两位客人。",
             "pause_after_ms": 800},
            {"voice": "zh_male_narrator", "text": "两人对视一眼，谁也没有说话。"},
        ]
        d = ok(c.post("/v1/dub", json={"lines": lines, "name": "smoke_dub", "format": "mp3", "srt_speaker": True}))
        r = d["result"]
        spans = ", ".join(f"{t['start']:.1f}-{t['end']:.1f}" for t in r["lines"])
        return f"{r['duration_s']}s [{spans}] {r['path']} warn={r['warnings']}"
    step("dub script (5 lines)", dub)

    def openai():
        r = c.post("/v1/audio/speech", json={"model": "breeze-tts-2", "input": "这是一个兼容接口测试。",
                                             "voice": "zh_female_narrator", "response_format": "mp3"})
        if r.status_code != 200:
            raise RuntimeError(r.text[:200])
        return f"{len(r.content)} bytes {r.headers['content-type']}"
    step("openai /v1/audio/speech", openai)

    def async_job():
        d = c.post("/v1/tts", json={"text": "异步任务测试。", "wait": False}).json()
        job = ok(c.get(f"/v1/jobs/{d['id']}", params={"wait_seconds": 120}))
        return f"{job['status']} {job['result']['duration_s']}s"
    step("async job + poll", async_job)

    for vid in ("smoke_villain", "smoke_clone"):
        c.delete(f"/v1/voices/{vid}")
    print("health:", c.get("/health").json())
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
