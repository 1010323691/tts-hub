"""Service tests with a fake engine (no GPU): text chunking, voices, tts/dub jobs, files, auth."""
from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from tts_hub import audio
from tts_hub.api import create_app
from tts_hub.config import Settings
from tts_hub.engine import Row, RowResult, plan_batches
from tts_hub.text import frame_cap, split_text, text_chars


class FakeEngine:
    device = "fake"
    modes = ("clone", "design")

    def __init__(self, name="breeze"):
        self.name = name
        self.calls: list[list[Row]] = []
        self.loaded = False

    def load(self):
        self.loaded = True
        self.loads = getattr(self, "loads", 0) + 1

    def unload(self):
        self.loaded = False

    def vram(self):
        return None

    def encode_reference(self, wav, sr, ref_text):
        return {"codes": "codes", "ref_text": ref_text, "engine": self.name}

    def synthesize(self, rows, progress=None):
        self.calls.append(rows)
        out = []
        for r in rows:  # 0.1 s of tone per char, padded with silence
            n = int(0.1 * audio.SAMPLE_RATE * text_chars(r.text))
            tone = 0.3 * np.sin(np.arange(n) / 10).astype(np.float32)
            wav = np.concatenate([np.zeros(4800, np.float32), tone, np.zeros(4800, np.float32)])
            out.append(RowResult(wav, int(wav.size / 1920), False))
        if progress:
            progress(len(rows), len(rows))
        return out


def _wav_bytes(seconds=3.0, sr=16000):
    buf = io.BytesIO()
    t = np.arange(int(seconds * sr)) / sr
    sf.write(buf, 0.3 * np.sin(2 * np.pi * 220 * t), sr, format="WAV")
    return buf.getvalue()


@pytest.fixture
def client(tmp_path):
    settings = Settings(data_dir=tmp_path, seed_builtin_voices=True)
    engine = FakeEngine()
    app = create_app(settings, engine)
    with TestClient(app) as c:
        manager = app.state.manager
        for _ in range(200):
            if manager.state == "ready":
                break
            import time
            time.sleep(0.02)
        c.engine = engine
        yield c


# -- pure helpers ---------------------------------------------------------------------------
def test_split_text_packs_sentences():
    text = "第一句话。第二句话！" * 20
    chunks = split_text(text, 30)
    assert all(text_chars(c) <= 30 for c in chunks)
    assert "".join(chunks) == text


def test_split_text_long_sentence_and_english():
    long = "，".join(["这是一个很长的分句"] * 20) + "。"
    assert all(text_chars(c) <= 40 for c in split_text(long, 40))
    en = "Hello there. This is a test. " * 10
    chunks = split_text(en, 50)
    assert all(text_chars(c) <= 50 for c in chunks) and chunks[0].startswith("Hello there.")


def test_frame_cap_bounds():
    assert frame_cap(1) == 150 and frame_cap(10_000) == 1500 and frame_cap(100) == 700


def test_plan_batches_groups_by_template_and_sorts():
    p = {"codes": 1, "ref_text": "x"}
    rows = [Row("长" * 30, p), Row("短", p), Row("设计", None, "温柔"), Row("中" * 10, p, "生气")]
    batches = plan_batches(rows, 16)
    assert sorted(map(sorted, batches)) == [[0, 1], [2], [3]]
    assert [b for b in batches if 0 in b][0] == [1, 0]  # shortest first
    assert len(plan_batches([Row("a", p)] * 5, 2)) == 3
    # a per-line seed must not be swallowed by a batch-mate's seed
    seeded = plan_batches([Row("a", p), Row("b", p, seed=7), Row("c", p)], 16)
    assert sorted(map(sorted, seeded)) == [[0, 2], [1]]


def test_concat_and_srt():
    full, spans = audio.concat([np.ones(24000, np.float32), np.ones(12000, np.float32)], [500, 0])
    assert full.size == 24000 + 12000 + 12000
    assert spans == [(0.0, 1.0), (1.5, 2.0)]
    srt = audio.to_srt([{"start": 0, "end": 1.0, "text": "你好", "speaker": "甲"}])
    assert "00:00:00,000 --> 00:00:01,000" in srt and "甲：你好" in srt


# -- API ------------------------------------------------------------------------------------
def test_health_and_builtin_voices(client):
    assert client.get("/health").json()["status"] == "ready"
    ids = {v["id"] for v in client.get("/v1/voices").json()["voices"]}
    assert {"zh_female_narrator", "en_male_narrator"} <= ids
    v = client.get("/v1/voices/zh_female_narrator").json()
    assert v["builtin"] and v["kind"] == "clone" and v["consistent"]


def test_clone_voice_lifecycle(client):
    r = client.post("/v1/voices", data={"voice_id": "hero", "name": "主角", "ref_text": "参考文本"},
                    files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 201, r.text
    assert r.json()["ref_seconds"] == pytest.approx(3.0, abs=0.01)
    dup = client.post("/v1/voices", data={"voice_id": "hero", "ref_text": "x"},
                      files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")})
    assert dup.status_code == 400
    assert client.get("/v1/voices/hero/audio").status_code == 200
    assert client.delete("/v1/voices/hero").status_code == 200
    assert client.get("/v1/voices/hero").status_code == 404
    assert client.delete("/v1/voices/zh_female_narrator").status_code == 400


def test_clone_voice_rejects_short_reference(client):
    r = client.post("/v1/voices", data={"voice_id": "short", "ref_text": "x"},
                    files={"ref_audio": ("ref.wav", _wav_bytes(0.3), "audio/wav")})
    assert r.status_code == 400


def test_design_voice_freeze_and_unfrozen(client):
    r = client.post("/v1/voices/design", json={"voice_id": "villain", "instruction": "阴冷低沉的男声"})
    assert r.status_code == 201, r.text
    assert r.json()["kind"] == "clone" and r.json()["source"] == "design"
    r = client.post("/v1/voices/design", json={"voice_id": "free", "instruction": "温柔", "freeze": False})
    assert r.status_code == 201 and r.json()["kind"] == "design"


def test_tts_sync_returns_file(client):
    r = client.post("/v1/tts", json={"text": "你好世界。今天天气很好。", "voice": "zh_male_narrator",
                                     "format": "mp3"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "done"
    res = body["result"]
    assert res["format"] == "mp3" and res["duration_s"] > 0 and res["url"].startswith("http")
    assert client.get(res["url"].replace("http://testserver", "")).status_code == 200


def test_tts_unknown_voice_404(client):
    assert client.post("/v1/tts", json={"text": "x", "voice": "nope"}).status_code == 404


def test_tts_async_then_poll(client):
    r = client.post("/v1/tts", json={"text": "异步任务。", "wait": False})
    assert r.status_code in (200, 202)
    job = client.get(f"/v1/jobs/{r.json()['id']}", params={"wait_seconds": 10}).json()
    assert job["status"] == "done"


def test_dub_script(client):
    lines = [
        {"voice": "zh_male_narrator", "text": "夜色渐深。"},
        {"voice": "zh_young_female", "text": "你听到了吗？", "instruction": "紧张地小声说"},
        {"voice": "zh_young_female", "text": "好像有人。", "pause_after_ms": 1000},
        {"voice": "zh_young_male", "text": "别怕，我在。", "speaker": "阿明"},
    ]
    r = client.post("/v1/dub", json={"lines": lines, "name": "第一集", "srt_speaker": True})
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    tl = res["lines"]
    assert len(tl) == 4 and all(t["end"] > t["start"] for t in tl)
    assert tl[1]["start"] - tl[0]["end"] == pytest.approx(0.4, abs=0.01)   # speaker change
    assert tl[2]["start"] - tl[1]["end"] == pytest.approx(0.25, abs=0.01)  # same speaker
    assert tl[3]["start"] - tl[2]["end"] == pytest.approx(1.0, abs=0.01)   # explicit pause
    srt = client.get(res["srt"]["url"].replace("http://testserver", "")).text
    assert "阿明：别怕，我在。" in srt
    assert res["path"].endswith("第一集.wav")
    # the instruction reaches only its own line
    rows = client.engine.calls[-1]
    assert [r.instruction for r in rows] == ["", "紧张地小声说", "", ""]


def test_openai_compatible(client):
    r = client.post("/v1/audio/speech", json={"input": "你好", "voice": "alloy", "response_format": "wav"})
    assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
    data, sr = sf.read(io.BytesIO(r.content))
    assert sr == 24000 and data.size > 0


def test_file_traversal_blocked(client):
    assert client.get("/files/../voices/zh_girl/voice.json").status_code == 404


def _wait_for(predicate, seconds=5.0):
    import time
    end = time.time() + seconds
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_idle_unload_reload_then_shutdown(tmp_path):
    settings = Settings(data_dir=tmp_path, seed_builtin_voices=False,
                        idle_unload_s=0.3, idle_shutdown_s=0.6, idle_check_s=0.05)
    engine = FakeEngine()
    app = create_app(settings, engine)
    manager = app.state.manager
    stopped = []
    manager.on_shutdown = lambda: stopped.append(True)
    with TestClient(app) as c:
        assert _wait_for(lambda: manager.state == "unloaded")
        assert not engine.loaded
        health = c.get("/health")
        assert health.status_code == 200 and health.json()["status"] == "unloaded"
        # health polling must not count as activity: the shutdown clock keeps running
        r = c.post("/v1/tts", json={"text": "重新加载。"})
        assert r.status_code == 200 and r.json()["status"] == "done"
        assert engine.loads == 2 and manager.state == "ready"
        assert _wait_for(lambda: stopped, 5)
        assert manager.state == "stopping" and not engine.loaded
        assert c.post("/v1/tts", json={"text": "x"}).status_code == 503


def test_manual_unload_and_disabled_idle(tmp_path):
    settings = Settings(data_dir=tmp_path, seed_builtin_voices=False,
                        idle_unload_s=0, idle_shutdown_s=0, idle_check_s=0.05)
    engine = FakeEngine()
    app = create_app(settings, engine)
    manager = app.state.manager
    with TestClient(app) as c:
        assert _wait_for(lambda: manager.state == "ready")
        import time
        time.sleep(0.3)
        assert manager.state == "ready"  # idle actions disabled
        assert c.post("/v1/unload").status_code == 200
        assert _wait_for(lambda: manager.state == "unloaded")
        assert c.post("/v1/tts", json={"text": "你好。"}).json()["status"] == "done"
        assert manager.state == "ready"


@pytest.fixture
def two_engines(tmp_path):
    engines = {"breeze": FakeEngine("breeze"), "qwen3": FakeEngine("qwen3")}
    app = create_app(Settings(data_dir=tmp_path, seed_builtin_voices=True), engines)
    with TestClient(app) as c:
        assert _wait_for(lambda: app.state.manager.state == "ready")
        c.engines = engines
        yield c


def test_presets_listed_and_protected(two_engines):
    c = two_engines
    voices = {v["id"]: v for v in c.get("/v1/voices").json()["voices"]}
    assert voices["qwen_vivian"]["kind"] == "preset" and voices["qwen_vivian"]["engines"] == ["qwen3"]
    assert voices["zh_female_narrator"]["engines"] == ["breeze", "qwen3"]
    assert c.delete("/v1/voices/qwen_vivian").status_code == 400
    r = c.post("/v1/voices", data={"voice_id": "qwen_vivian", "ref_text": "x"},
               files={"ref_audio": ("ref.wav", _wav_bytes(), "audio/wav")})
    assert r.status_code == 400


def test_preset_routes_to_qwen3(two_engines):
    c = two_engines
    r = c.post("/v1/tts", json={"text": "预置音色。", "voice": "qwen_uncle_fu", "instruction": "开心"})
    assert r.status_code == 200, r.text
    assert r.json()["result"]["engine"] == "qwen3"
    row = c.engines["qwen3"].calls[-1][0]
    assert row.speaker == "uncle_fu" and row.instruction == "开心" and row.language == ""
    c.post("/v1/tts", json={"text": "Hello.", "voice": "qwen_ryan", "language": "en"})
    assert c.engines["qwen3"].calls[-1][0].language == "english"
    assert not c.engines["qwen3"].calls[-1][0].prompt
    bad = c.post("/v1/tts", json={"text": "x", "voice": "qwen_uncle_fu", "engine": "breeze"})
    assert bad.status_code == 400 and "qwen3" in bad.json()["detail"]


def test_clone_prompt_cached_per_engine_and_qwen_instruction_warning(two_engines):
    c = two_engines
    r = c.post("/v1/tts", json={"text": "克隆。", "voice": "zh_male_narrator", "engine": "qwen3",
                                "instruction": "生气地说"})
    res = r.json()["result"]
    assert res["engine"] == "qwen3" and any("instruction" in w for w in res["warnings"])
    assert c.engines["qwen3"].calls[-1][0].prompt["engine"] == "qwen3"
    c.post("/v1/tts", json={"text": "克隆。", "voice": "zh_male_narrator"})
    assert c.engines["breeze"].calls[-1][0].prompt["engine"] == "breeze"


def test_dub_mixes_engines(two_engines):
    c = two_engines
    lines = [{"voice": "zh_male_narrator", "text": "旁白第一句。"},
             {"voice": "qwen_vivian", "text": "预置音色说话。"},
             {"voice": "zh_young_female", "text": "这一行指定千问。", "engine": "qwen3"},
             {"voice": "zh_male_narrator", "text": "旁白第二句。"}]
    r = c.post("/v1/dub", json={"lines": lines})
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert [t["engine"] for t in res["lines"]] == ["breeze", "qwen3", "qwen3", "breeze"]
    assert res["engines"] == ["breeze", "qwen3"]
    bad = c.post("/v1/dub", json={"lines": [{"voice": "qwen_sohee", "text": "x", "engine": "breeze"}]})
    assert bad.status_code == 400 and "第 1 行" in bad.json()["detail"]


def test_qwen3_engine_loads_on_demand_and_unloads_with_breeze(tmp_path):
    engines = {"breeze": FakeEngine("breeze"), "qwen3": FakeEngine("qwen3")}
    app = create_app(Settings(data_dir=tmp_path, seed_builtin_voices=False), engines)
    manager = app.state.manager
    with TestClient(app) as c:
        assert _wait_for(lambda: manager.state == "ready")
        assert engines["breeze"].loaded and not engines["qwen3"].loaded
        assert c.get("/health").json()["engines"]["qwen3"]["loaded"] is False
        c.post("/v1/tts", json={"text": "按需加载。", "engine": "qwen3"})
        assert engines["qwen3"].loaded
        c.post("/v1/unload")
        assert _wait_for(lambda: manager.state == "unloaded")
        assert not engines["breeze"].loaded and not engines["qwen3"].loaded


def test_qwen_engine_helpers():
    from tts_hub.jobs import qwen_language
    from tts_hub.qwen_engine import frame_cap as qcap
    from tts_hub.qwen_engine import qwen_batch_key
    assert qwen_language("zh") == "chinese" and qwen_language("auto") == "" and qwen_language("English") == "english"
    assert qcap(1) == 128 and qcap(100) == 450 and qcap(10_000) == 2048
    p = object()
    rows = [Row("a", speaker="vivian"), Row("b", p), Row("c", None, "温柔"), Row("d", speaker="ryan", seed=7)]
    assert sorted(map(sorted, plan_batches(rows, 16, key=qwen_batch_key))) == [[0], [1], [2], [3]]
    rows = [Row("a", speaker="vivian"), Row("bb", speaker="ryan", instruction="开心")]
    assert plan_batches(rows, 16, key=qwen_batch_key) == [[0, 1]]  # speakers/instructs share a batch


def test_qwen3_vram_model_matches_narrify():
    from tts_hub import batch_limits as bl
    from tts_hub.qwen_engine import frame_cap as qcap

    def cap(model, chars, budget):
        return model.cap(bl.PRODUCTION_PROMPT_FRAMES + chars + qcap(chars), budget, max_rows=10_000)

    # NarrifyAudio backend/core/tts_batch_limits.py: AUTO_BATCH_POINTS and auto_batch_cap outputs
    ref = 28_774_845_440
    assert [cap(bl.QWEN3, c, ref) for c in (5, 10, 20, 40, 80, 120, 160, 200)] == \
        [632, 622, 604, 500, 326, 242, 192, 160]
    assert [cap(bl.QWEN3, c, 20 * 2**30) for c in (5, 40, 200)] == [472, 372, 118]
    assert cap(bl.QWEN3_UNTRIMMED, 80, ref) == 170


def test_breeze_vram_model_covers_calibration_samples():
    """Every measured batch (scripts/calibrate_vram.py on the RTX 5090) is at or below the model."""
    from tts_hub import batch_limits as bl
    # (mode, rows, frames, measured peak GiB) — a subset of data/vram_fit.json's samples
    samples = [("plain", 1, 539, 1.03), ("plain", 96, 581, 6.24), ("plain", 48, 322, 1.79),
               ("plain", 96, 180, 2.27), ("cfg", 1, 488, 1.04), ("cfg", 96, 512, 10.39),
               ("cfg", 48, 245, 2.50), ("cfg", 96, 66, 1.28)]
    for mode, rows, frames, gib in samples:
        model = bl.BREEZE_CFG if mode == "cfg" else bl.BREEZE_PLAIN
        assert model.predicted_bytes(rows, frames) >= gib * 2**30 * 0.995, (mode, rows, frames)
        assert model.predicted_bytes(rows, frames) <= gib * 2**30 * 1.7


def test_vram_plan_respects_budget_and_fallback():
    from tts_hub import batch_limits as bl
    model = bl.VramModel("t", row_fixed_bytes=0, bytes_per_row_frame=1000, headroom=1.0)
    chars = {i: c for i, c in enumerate([10, 10, 10, 10, 50, 50])}
    plan = bl.plan(list(chars), lambda p: 0, chars.get, lambda c: 0, model, budget=40_000, max_rows=128)
    # 10-char rows: 4000 rows fit -> one batch; a 50-char row lifts the batch to 50k/row -> split
    assert plan == [[0, 1, 2, 3], [4], [5]]
    assert all(len(b) <= 4 for b in bl.plan(list(chars), lambda p: 0, chars.get, lambda c: 0, model, None))
    assert bl.oom_retry_rows(10) == 9 and bl.oom_retry_rows(2) == 1 and bl.oom_retry_rows(1) == 1


def test_run_batches_splits_on_oom():
    from tts_hub.engine import run_batches

    class OutOfMemoryError(RuntimeError):
        pass

    calls, ooms = [], []

    def generate(part):
        calls.append(len(part))
        if len(part) > 4:
            raise OutOfMemoryError("CUDA out of memory")
        return [RowResult(np.zeros(10, np.float32), 1, False) for _ in part]

    rows = [Row(str(i)) for i in range(10)]
    res = run_batches([list(range(10))], rows, generate, on_oom=lambda: ooms.append(1))
    assert all(r is not None for r in res) and len(ooms) >= 1
    assert calls[0] == 10 and calls[1] == 9  # shortest 90 % retried first
    with pytest.raises(OutOfMemoryError):
        run_batches([[0, 1, 2, 3, 4]], rows, lambda part: (_ for _ in ()).throw(OutOfMemoryError("x")))


def test_batch_endpoint(client):
    items = [{"id": "greet", "text": "你好。", "voice": "zh_girl"},
             {"text": "第二条，没有 id。", "voice": "zh_old_man", "instruction": "慢慢说"},
             {"id": "bad/../name", "text": "第三条。"}]
    r = client.post("/v1/batch", json={"items": items, "format": "mp3"})
    assert r.status_code == 200, r.text
    res = r.json()["result"]
    assert res["count"] == 3 and [i["id"] for i in res["items"]] == ["greet", "002", "bad_name"]
    for item in res["items"]:
        assert item["file"]["path"].endswith(f"{item['id']}.mp3") and item["duration_s"] > 0
    assert client.get(res["manifest"]["url"].replace("http://testserver", "")).status_code == 200
    dup = client.post("/v1/batch", json={"items": [{"id": "a", "text": "x"}, {"id": "a", "text": "y"}]})
    assert dup.status_code == 422


def test_api_key(tmp_path):
    app = create_app(Settings(data_dir=tmp_path, api_key="s3cret", seed_builtin_voices=False), FakeEngine())
    with TestClient(app) as c:
        assert c.get("/health").status_code in (200, 503)
        assert c.get("/v1/voices").status_code == 401
        assert c.get("/v1/voices", headers={"Authorization": "Bearer s3cret"}).status_code == 200


def test_env_names_fall_back_to_breeze_prefix(monkeypatch):
    for name in ("TTS_PORT", "BREEZE_PORT", "TTS_MAX_BATCH", "BREEZE_MAX_BATCH", "BREEZE_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("BREEZE_PORT", "7999")       # pre-rename name still works
    monkeypatch.setenv("BREEZE_MAX_BATCH", "8")
    monkeypatch.setenv("TTS_MAX_BATCH", "16")       # new name wins when both are set
    monkeypatch.setenv("BREEZE_MODEL", "local/breeze")  # engine-specific name unchanged
    s = Settings.from_env()
    assert (s.port, s.max_batch, s.model) == (7999, 16, "local/breeze")
