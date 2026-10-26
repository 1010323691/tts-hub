"""Large-batch run against a running service: auto batch sizes, VRAM peak, throughput.

    .venv\\Scripts\\python.exe scripts\\batch_stress_test.py [--items 300] [--engine breeze|qwen3|both]
"""
from __future__ import annotations

import argparse
import random
import subprocess
import sys
import threading
import time

import httpx

SENTENCES = [
    "好的。", "你来了？", "快跑，别回头！", "今天的天气真不错。", "这件事我们明天再商量吧。",
    "夜色渐深，古堡里传来一阵奇怪的声响，大家都屏住了呼吸。",
    "他推开那扇沉重的木门，灰尘在阳光里飞舞，屋子里静得能听见自己的心跳。",
    "小镇的清晨总是从面包店开始的，新鲜出炉的香气飘过街道，叫醒了还在睡梦中的人们。",
    "多年以后，当她再次站在这座桥上，望着桥下缓缓流过的河水，才终于明白那年夏天父亲说过的话，"
    "原来人生中最重要的东西，往往是在失去之后才看得清楚。",
]


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout
    return int(out.split()[0])


def run(c: httpx.Client, engine: str, voices: list[str], n: int) -> None:
    rnd = random.Random(7)
    items = [{"id": f"{engine}_{i:04d}", "text": rnd.choice(SENTENCES), "voice": voices[i % len(voices)]}
             for i in range(n)]
    chars = sum(len(x["text"]) for x in items)
    peak = [gpu_used_mib()]
    stop = threading.Event()

    def sample():
        while not stop.is_set():
            peak[0] = max(peak[0], gpu_used_mib())
            time.sleep(0.5)

    th = threading.Thread(target=sample, daemon=True)
    th.start()
    t = time.time()
    r = c.post("/v1/batch", json={"items": items, "engine": engine, "save_lines": False, "wait_seconds": 3600})
    el = time.time() - t
    stop.set()
    th.join()
    body = r.json()
    if body.get("status") != "done":
        print(f"[FAIL] {engine}: {r.status_code} {body.get('error') or body}")
        return
    res = body["result"]
    h = c.get("/health").json()["engines"][engine]
    batches = h.get("last_batches", [])
    print(f"[OK]   {engine}: {n} 条 / {chars} 字  {el:.1f}s → {res['total_duration_s']:.0f}s 音频 "
          f"（{res['total_duration_s'] / el:.1f}× 实时）  GPU 峰值 {peak[0] / 1024:.1f} GiB  "
          f"批次 {len(batches)} 个：{batches}  oom_scale={h.get('oom_scale', 1.0)}  warn={len(res['warnings'])}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:7861")
    ap.add_argument("--items", type=int, default=300)
    ap.add_argument("--engine", default="both")
    args = ap.parse_args()
    c = httpx.Client(base_url=args.url, timeout=3700)
    if args.engine in ("breeze", "both"):
        run(c, "breeze", ["zh_male_narrator", "zh_young_female", "zh_young_male", "zh_old_man"], args.items)
    if args.engine in ("qwen3", "both"):
        run(c, "qwen3", ["qwen_vivian", "qwen_serena", "qwen_uncle_fu", "qwen_eric"], args.items)
    return 0


if __name__ == "__main__":
    sys.exit(main())
