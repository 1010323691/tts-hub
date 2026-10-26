"""MCP (stdio) bridge: exposes the TTS Hub HTTP service (Breeze TTS 2 + Qwen3-TTS) as agent tools.

The model stays loaded in the HTTP service; this process only forwards calls, uploads local
reference audio and copies results to the paths the agent asks for.

    claude mcp add tts-hub -- "<project>\\.venv\\Scripts\\python.exe" "<project>\\mcp_server.py"

Environment: TTS_URL (default http://127.0.0.1:7861), TTS_API_KEY, TTS_MCP_TIMEOUT; the older
BREEZE_* names are still read as fallbacks.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from mcp.server.fastmcp import FastMCP

def _env(name: str, default: str = "") -> str:
    return os.environ.get(f"TTS_{name}") or os.environ.get(f"BREEZE_{name}") or default


BASE = _env("URL", "http://127.0.0.1:7861").rstrip("/")
KEY = _env("API_KEY")
TIMEOUT = float(_env("MCP_TIMEOUT", "3600"))

mcp = FastMCP("tts-hub", instructions=(
    "本地统一 TTS 调度服务（配音），两个引擎：breeze（Breeze TTS 2，中英文，克隆音色也能用 instruction 控制语气；权重仅限非商业）"
    "与 qwen3（Qwen3-TTS，Apache-2.0 可商用，10 种语言，qwen_* 预置音色支持 instruction，克隆音色不支持 instruction）。"
    "流程：tts_list_voices 选音色（engines 字段表示可用引擎）→ 需要新角色时用 tts_design_voice（文字描述）"
    "或 tts_clone_voice（参考音频）建音色 → 单句用 tts_speak，多角色剧本用 tts_dub（返回合并音频、逐行音频与 SRT 字幕），"
    "大量互相独立的条目用 tts_batch（每条一个文件）。dub/batch 一次提交全部内容，服务按显存自动分批并行，比逐条调用快得多。"
    "engine 参数选引擎，缺省用服务默认引擎，qwen_* 预置音色自动走 qwen3。breeze 的文本可插入声音事件："
    "中文 [笑] [叹气] [咳嗽] [清嗓子]，英文 (laugh) (sigh) (cough)。"))


def _client() -> httpx.Client:
    headers = {"Authorization": f"Bearer {KEY}"} if KEY else {}
    return httpx.Client(base_url=BASE, headers=headers, timeout=TIMEOUT)


def _check(r: httpx.Response) -> Any:
    if r.status_code >= 400:
        try:
            detail = r.json()
        except ValueError:
            detail = r.text
        raise RuntimeError(f"TTS 服务返回 HTTP {r.status_code}: {detail}")
    return r.json()


def _wait(c: httpx.Client, body: dict) -> dict:
    """Follow a 202 (still running) response until the job finishes."""
    while body.get("status") in ("queued", "running"):
        body = _check(c.get(f"/v1/jobs/{body['id']}", params={"wait_seconds": 300}))
    if body.get("status") != "done":
        raise RuntimeError(f"任务 {body.get('id')} {body.get('status')}: {body.get('error')}")
    return body


def _save(c: httpx.Client, url: str, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with c.stream("GET", url) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes():
                f.write(chunk)
    return str(dest.resolve())


@mcp.tool()
def tts_health() -> dict:
    """服务状态。ready 可合成；unloaded 也可直接提交（权重已因空闲卸载，首个请求自动重载、多等约 30 秒）；
    loading/seeding/warming 正在启动；连不上表示服务未启动或已因空闲自动退出，需要运行 start.ps1。"""
    with _client() as c:
        r = c.get("/health")
        return r.json()


@mcp.tool()
def tts_list_voices() -> list[dict]:
    """列出可用音色（id、名称、语言、描述、engines 可用引擎）。consistent=true 的音色每次合成音色一致，
    适合同一角色多句台词。kind=preset 是 Qwen3 预置音色（只能用 qwen3 引擎）。"""
    with _client() as c:
        voices = _check(c.get("/v1/voices"))["voices"]
    return [{k: v.get(k) for k in ("id", "name", "kind", "language", "engines", "consistent", "builtin",
                                   "description") if v.get(k) not in (None, "")} for v in voices]


@mcp.tool()
def tts_design_voice(voice_id: str, instruction: str, name: str = "", language: str = "",
                        sample_text: str = "", engine: str = "", seed: int = 42,
                        overwrite: bool = False) -> dict:
    """用文字描述创建新音色并固化（生成一段样本作为参考音频，之后每句音色一致，两个引擎都能用）。

    instruction 示例："一位阴险狡诈的中年男性，声音低沉沙哑，语速缓慢，带着冷笑。"
    英文角色用英文描述并设 language="en"。voice_id 只能含字母数字下划线连字符，不能以 qwen_ 预置名冲突。
    engine：用哪个引擎生成样本（breeze 或 qwen3，缺省为默认引擎）；不满意可换 seed 并 overwrite=true 重建。
    """
    with _client() as c:
        body = _check(c.post("/v1/voices/design", json={
            "voice_id": voice_id, "name": name, "instruction": instruction, "language": language,
            "sample_text": sample_text or None, "engine": engine or None, "seed": seed,
            "overwrite": overwrite}))
    return body


@mcp.tool()
def tts_clone_voice(voice_id: str, ref_audio_path: str, ref_text: str, name: str = "",
                       instruction: str = "", overwrite: bool = False) -> dict:
    """用本地参考音频克隆音色。ref_audio_path：本机音频文件（wav/mp3/flac/ogg/m4a，推荐 5–15 秒清晰人声）；
    ref_text：该音频的逐字文本（必须准确）。instruction：可选，作为该音色的默认语气指导。"""
    path = Path(ref_audio_path)
    if not path.is_file():
        raise FileNotFoundError(ref_audio_path)
    with _client() as c, open(path, "rb") as f:
        return _check(c.post("/v1/voices", data={
            "voice_id": voice_id, "name": name, "ref_text": ref_text, "instruction": instruction,
            "overwrite": str(overwrite).lower()}, files={"ref_audio": (path.name, f)}))


@mcp.tool()
def tts_delete_voice(voice_id: str) -> dict:
    """删除自建音色（内置音色不可删除）。"""
    with _client() as c:
        return _check(c.delete(f"/v1/voices/{voice_id}"))


@mcp.tool()
def tts_speak(text: str, voice: str = "", instruction: str = "", format: str = "wav",
               output_path: str = "", seed: int = 42, engine: str = "", language: str = "") -> dict:
    """合成一段语音（长文本自动按句切分后拼接）。

    voice：音色 id（缺省 zh_female_narrator）。instruction：语气/情绪/语速，如"兴奋地，语速稍快"。
    engine：breeze 或 qwen3（缺省为默认引擎；qwen_* 预置音色自动用 qwen3）。
    language：仅 qwen3，zh/en/ja/ko/de/fr/es/it/pt/ru，缺省自动识别。
    format：wav/mp3/flac/ogg。output_path：保存到本机该路径（缺省只返回服务端文件路径）。
    返回 file（本机文件路径）、duration_s、url、engine。
    """
    with _client() as c:
        body = _wait(c, _check(c.post("/v1/tts", json={
            "text": text, "voice": voice or None, "instruction": instruction, "format": format,
            "seed": seed, "engine": engine or None, "language": language, "wait_seconds": 600})))
        res = body["result"]
        file = _save(c, res["url"], Path(output_path)) if output_path else res["path"]
    return {"file": file, "duration_s": res["duration_s"], "url": res["url"], "voice": res["voice"],
            "engine": res.get("engine"), "warnings": res.get("warnings", [])}


@mcp.tool()
def tts_dub(lines: list[dict], name: str = "dub", format: str = "wav", output_dir: str = "",
               pause_ms: int = 400, same_speaker_pause_ms: int = 250, srt_speaker: bool = False,
               seed: int = 42, engine: str = "", language: str = "") -> dict:
    """多角色配音：按顺序合成剧本台词并拼接，同时输出逐行音频、SRT 字幕和时间轴。

    lines：[{"voice": "zh_male_narrator", "text": "台词", "instruction": "可选语气",
             "speaker": "可选字幕角色名", "pause_after_ms": 可选该行后停顿, "seed": 可选该行种子,
             "engine": 可选该行引擎, "language": 可选该行语言}]
    engine / language：整份剧本的默认引擎与语言（同一剧本可混用两个引擎）。
    pause_ms：换角色停顿；same_speaker_pause_ms：同角色连续停顿。
    output_dir：把合并音频、字幕、逐行音频复制到本机该目录（缺省返回服务端路径）。
    """
    with _client() as c:
        body = _wait(c, _check(c.post("/v1/dub", json={
            "lines": lines, "name": name, "format": format, "pause_ms": pause_ms,
            "same_speaker_pause_ms": same_speaker_pause_ms, "srt_speaker": srt_speaker, "seed": seed,
            "engine": engine or None, "language": language, "wait_seconds": 600})))
        res = body["result"]
        out = {"file": res["path"], "srt": res["srt"]["path"], "output_dir": res["output_dir"],
               "duration_s": res["duration_s"], "warnings": res.get("warnings", [])}
        if output_dir:
            dest = Path(output_dir)
            out["file"] = _save(c, res["url"], dest / Path(res["path"]).name)
            out["srt"] = _save(c, res["srt"]["url"], dest / Path(res["srt"]["path"]).name)
            _save(c, res["timeline_json"]["url"], dest / "timeline.json")
            out["output_dir"] = str(dest.resolve())
            for t in res["lines"]:
                if t.get("file"):
                    t["file"]["path"] = _save(c, t["file"]["url"], dest / "lines" / Path(t["file"]["path"]).name)
        out["timeline_json"] = str(Path(out["output_dir"]) / "timeline.json")
        out["engines"] = res.get("engines", [])
        out["lines"] = [{"index": t["index"], "voice": t["voice"], "engine": t.get("engine"),
                         "text": t["text"], "start": t["start"],
                         "end": t["end"], "duration_s": t["duration_s"], "capped": t["capped"],
                         "file": (t.get("file") or {}).get("path")} for t in res["lines"]]
    return out


@mcp.tool()
def tts_batch(items: list[dict], format: str = "wav", output_dir: str = "", engine: str = "",
                 language: str = "", seed: int = 42) -> dict:
    """批量合成互相独立的语音条目：每条一个文件，不拼接、不出字幕（游戏/应用台词、提示音、单词朗读、
    多版本试听等）。一次提交全部条目，服务按显存自动分批并行生成，比逐条调 tts_speak 快很多倍。

    items：[{"id": "可选文件名", "text": "内容", "voice": "音色", "instruction": "可选语气",
             "engine": 可选, "language": 可选, "seed": 可选}]，最多 2000 条。
    output_dir：把所有文件与 manifest.json 复制到本机该目录（缺省返回服务端路径）。
    返回每条的 id、file、duration_s、capped，以及 manifest 路径。
    """
    with _client() as c:
        body = _wait(c, _check(c.post("/v1/batch", json={
            "items": items, "format": format, "seed": seed, "engine": engine or None,
            "language": language, "wait_seconds": 600})))
        res = body["result"]
        out = {"output_dir": res["output_dir"], "count": res["count"], "engines": res.get("engines", []),
               "total_duration_s": res["total_duration_s"], "manifest": res["manifest"]["path"],
               "warnings": res.get("warnings", [])}
        if output_dir:
            dest = Path(output_dir)
            for item in res["items"]:
                item["file"]["path"] = _save(c, item["file"]["url"], dest / Path(item["file"]["path"]).name)
            out["manifest"] = _save(c, res["manifest"]["url"], dest / "manifest.json")
            out["output_dir"] = str(dest.resolve())
        out["items"] = [{"id": i["id"], "voice": i["voice"], "engine": i["engine"], "duration_s": i["duration_s"],
                         "capped": i["capped"], "file": i["file"]["path"]} for i in res["items"]]
    return out


@mcp.tool()
def tts_job(job_id: str) -> dict:
    """查询任务状态与结果（用于超时后继续等待长任务）。"""
    with _client() as c:
        return _check(c.get(f"/v1/jobs/{job_id}"))


if __name__ == "__main__":
    mcp.run()
