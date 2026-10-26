"""HTTP API (FastAPI). See AGENT_GUIDE.md / GET /v1/guide for the caller-facing contract."""
from __future__ import annotations

import asyncio
import logging
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from pydantic import BaseModel, Field, field_validator

from . import __version__, audio
from .config import PROJECT_ROOT, Settings
from .engine import BreezeEngine
from .jobs import TERMINAL, Job, JobManager
from .qwen_engine import Qwen3Engine
from .voices import VoiceError, VoiceStore, public_view, slugify

log = logging.getLogger("tts_hub.api")

Format = Literal["wav", "mp3", "flac", "ogg"]
MAX_TEXT_CHARS = 20000
MAX_LINES = 2000
OPENAI_VOICES = {"alloy", "ash", "ballad", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer", "verse"}
GUIDE = PROJECT_ROOT / "AGENT_GUIDE.md"
OPEN_PATHS = ("/health", "/", "/docs", "/openapi.json", "/redoc", "/v1/guide")


# -- request models ------------------------------------------------------------------------
Engine = Literal["breeze", "qwen3"]


class SynthOptions(BaseModel):
    format: Format = "wav"
    seed: int = 42
    engine: Engine | None = Field(None, description="breeze | qwen3；缺省为服务默认引擎（预置音色自动用 qwen3）")
    language: str = Field("", max_length=20,
                          description="仅 qwen3：zh/en/ja/ko/de/fr/es/it/pt/ru 或全称，缺省自动识别")
    cfg_scale: float | None = Field(None, gt=0, le=10,
                                    description="仅 breeze。默认：有 instruction 时 4，否则 1")
    trim_silence: bool = True
    normalize: bool = True
    chunk_pause_ms: int = Field(200, ge=0, le=5000, description="长文本切块之间的停顿")
    wait: bool = Field(True, description="true：同步等待结果；false：立即返回 job id")
    wait_seconds: float = Field(600, ge=1, le=3600, description="同步等待上限，超时返回 202 + job")


class TTSRequest(SynthOptions):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    voice: str | None = Field(None, description="音色 id（GET /v1/voices），缺省为 zh_female_narrator")
    instruction: str = Field("", max_length=1000,
                             description="语气/情绪/语速指导（Voice Direction）；对设计音色则追加到音色描述")
    split: bool = True


class DubLine(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    voice: str | None = None
    instruction: str = Field("", max_length=1000)
    speaker: str | None = Field(None, description="字幕里的角色名，缺省为音色名")
    pause_after_ms: int | None = Field(None, ge=0, le=60000)
    seed: int | None = Field(None, description="该行单独的随机种子（重做某一行时改它）")
    cfg_scale: float | None = Field(None, gt=0, le=10)
    engine: Engine | None = Field(None, description="该行单独指定引擎")
    language: str | None = Field(None, max_length=20)


class DubRequest(SynthOptions):
    lines: list[DubLine] = Field(..., min_length=1, max_length=MAX_LINES)
    name: str = Field("dub", max_length=64, description="输出文件名（不含扩展名）")
    pause_ms: int = Field(400, ge=0, le=60000, description="换角色时的停顿")
    same_speaker_pause_ms: int = Field(250, ge=0, le=60000, description="同角色连续两行的停顿")
    lead_silence_ms: int = Field(0, ge=0, le=60000)
    save_lines: bool = True
    srt_speaker: bool = Field(False, description="字幕行前加「角色名：」")

    @field_validator("name")
    @classmethod
    def _safe_name(cls, v: str) -> str:
        return slugify(v)


class BatchItem(BaseModel):
    id: str | None = Field(None, max_length=64, description="输出文件名（不含扩展名），缺省为序号")
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    voice: str | None = None
    instruction: str = Field("", max_length=1000)
    seed: int | None = None
    cfg_scale: float | None = Field(None, gt=0, le=10)
    engine: Engine | None = None
    language: str | None = Field(None, max_length=20)


class BatchRequest(SynthOptions):
    items: list[BatchItem] = Field(..., min_length=1, max_length=MAX_LINES,
                                   description="互相独立的语音条目，每条输出一个文件")

    @field_validator("items")
    @classmethod
    def _ids(cls, items: list[BatchItem]) -> list[BatchItem]:
        width = max(3, len(str(len(items))))
        seen: set[str] = set()
        for i, item in enumerate(items):
            item.id = slugify(item.id) if item.id else f"{i + 1:0{width}d}"
            if item.id in seen:
                raise ValueError(f"items 里 id 重复：{item.id}")
            seen.add(item.id)
        return items


class DesignVoiceRequest(BaseModel):
    voice_id: str | None = None
    name: str = ""
    instruction: str = Field(..., min_length=1, max_length=1000)
    sample_text: str | None = Field(None, max_length=300,
                                    description="用来固化音色的样本台词；缺省按语言自动选择")
    language: str = ""
    description: str = ""
    freeze: bool = Field(True, description="true：生成样本并存为参考音频（音色稳定）；false：每次按描述重新设计")
    engine: Engine | None = Field(None, description="用哪个引擎设计样本（qwen3 用 VoiceDesign 模型）")
    seed: int = 42
    overwrite: bool = False


class OpenAISpeech(BaseModel):
    model: str = Field("tts-1", description="含 qwen 走 qwen3、含 breeze 走 breeze，其余按音色/默认引擎")
    input: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    voice: str = "zh_female_narrator"
    instructions: str = ""
    response_format: Literal["wav", "mp3", "flac", "opus", "pcm", "aac"] = "mp3"
    speed: float = 1.0


# -- app -----------------------------------------------------------------------------------
def build_engines(settings: Settings) -> dict:
    return {
        "breeze": BreezeEngine(settings.model, settings.code_dir, settings.device, settings.max_batch,
                               settings.vram_fit),
        "qwen3": Qwen3Engine(settings.qwen_custom_model, settings.qwen_base_model,
                             settings.qwen_design_model, settings.device, settings.max_batch,
                             settings.vram_fit),
    }


def create_app(settings: Settings | None = None, engines=None) -> FastAPI:
    """``engines``: {name: engine}; a single engine object is registered as the default engine."""
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    voices = VoiceStore(settings.voices_dir)
    if engines is None:
        engines = build_engines(settings)
    elif not isinstance(engines, dict):
        engines = {settings.default_engine: engines}
    manager = JobManager(settings, voices, engines)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        manager.start()
        yield
        manager.stop()

    app = FastAPI(title="TTS Hub", version=__version__, lifespan=lifespan,
                  description="Breeze TTS 2 / Qwen3-TTS 配音服务：音色库、单句合成、多角色配音脚本、OpenAI 兼容接口。")
    app.state.settings, app.state.voices, app.state.manager = settings, voices, manager

    @app.middleware("http")
    async def auth(request: Request, call_next):
        if request.url.path not in OPEN_PATHS:
            manager.touch()  # health polling and docs never keep the model loaded
        if settings.api_key and request.url.path not in OPEN_PATHS:
            token = request.headers.get("x-api-key") or ""
            header = request.headers.get("authorization") or ""
            if header.lower().startswith("bearer "):
                token = header[7:].strip()
            if not secrets.compare_digest(token, settings.api_key):
                return JSONResponse({"detail": "未授权：需要 Authorization: Bearer <TTS_API_KEY>"},
                                    status_code=401)
        return await call_next(request)

    def base_url(request: Request) -> str:
        return settings.public_url or str(request.base_url).rstrip("/")

    def absolutize(value, request: Request):
        """Turn every ``url`` field of a result into an absolute URL."""
        if isinstance(value, dict):
            return {k: (base_url(request) + v if k == "url" and isinstance(v, str) and v.startswith("/")
                        else absolutize(v, request)) for k, v in value.items()}
        if isinstance(value, list):
            return [absolutize(v, request) for v in value]
        return value

    def job_view(job: Job, request: Request) -> dict:
        view = absolutize(job.view(), request)
        view["poll_url"] = f"{base_url(request)}/v1/jobs/{job.id}"
        if job.status == "queued":
            view["queue_position"] = manager.queue_position(job)
        return view

    def require_ready():
        # a failed default engine does not block the others; its own jobs fail with the reason
        if manager.state == "stopping":
            raise HTTPException(503, "服务正在关闭，请重新启动后再试")

    async def finish_or_accept(job: Job, wait: bool, wait_seconds: float, request: Request):
        if wait:
            await asyncio.to_thread(job.event.wait, wait_seconds)
        if job.status not in TERMINAL:
            return JSONResponse(job_view(job, request), status_code=202)
        if job.status == "error":
            return JSONResponse(job_view(job, request), status_code=422)
        return job_view(job, request)

    def check_voice(voice_id: str | None, engine: str | None = None, where: str = ""):
        voice = voices.get(voice_id) if voice_id else None
        if voice_id and voice is None:
            raise HTTPException(404, f"{where}音色不存在：{voice_id}（GET /v1/voices 查看可用音色）")
        try:
            manager.resolve_engine(engine, voice)
        except VoiceError as exc:
            raise HTTPException(400, f"{where}{exc}") from exc

    # -- meta -------------------------------------------------------------------------------
    @app.get("/")
    def root(request: Request):
        return {"service": "tts-hub", "version": __version__, "state": manager.state,
                "docs": f"{base_url(request)}/docs", "guide": f"{base_url(request)}/v1/guide"}

    @app.get("/health")
    def health():
        default = manager.engines[manager.default_engine]
        body = {"status": manager.state, "default_engine": manager.default_engine,
                "engines": manager.engines_view(),
                "device": str(default.device) if getattr(default, "device", None) else None,
                "max_batch": settings.max_batch, "jobs": manager.stats(),
                "vram": manager.vram(), "idle": manager.idle_view()}
        if manager.state == "unloaded":
            body["note"] = "权重已因空闲卸载，下一个合成请求会自动重新加载（首个请求多等约 30 秒）"
        if manager.state_error:
            body["error"] = manager.state_error
        # unloaded still accepts work, so it counts as healthy
        return JSONResponse(body, status_code=200 if manager.state in ("ready", "unloaded") else 503)

    @app.post("/v1/unload")
    def unload_now():
        """立即卸载模型权重释放显存（下一个合成请求自动重新加载）。"""
        manager.request("unload")
        return {"requested": "unload", "state": manager.state}

    @app.post("/v1/shutdown")
    def shutdown_now():
        """排在已提交的任务之后关闭服务进程。"""
        manager.request("shutdown")
        return {"requested": "shutdown", "state": manager.state}

    @app.get("/v1/guide", response_class=PlainTextResponse)
    def guide():
        return PlainTextResponse(GUIDE.read_text(encoding="utf-8") if GUIDE.is_file() else "",
                                 media_type="text/markdown; charset=utf-8")

    # -- voices -----------------------------------------------------------------------------
    @app.get("/v1/voices")
    def list_voices():
        return {"voices": [public_view(v) for v in voices.list()]}

    @app.get("/v1/voices/{voice_id}")
    def get_voice(voice_id: str):
        voice = voices.get(voice_id)
        if voice is None:
            raise HTTPException(404, f"音色不存在：{voice_id}")
        return public_view(voice)

    @app.get("/v1/voices/{voice_id}/audio")
    def voice_audio(voice_id: str):
        voice = voices.get(voice_id)
        if voice is None or voice["kind"] != "clone":
            raise HTTPException(404, "该音色没有参考音频")
        return FileResponse(voices.reference_path(voice), media_type="audio/wav")

    @app.post("/v1/voices", status_code=201)
    async def create_clone_voice(
        request: Request,
        ref_text: str = Form(..., description="参考音频的逐字文本"),
        ref_audio: UploadFile | None = File(None, description="参考音频 wav/mp3/flac/ogg/m4a，推荐 5–15s"),
        ref_audio_path: str = Form("", description="服务端本机路径（仅限 localhost 调用）"),
        voice_id: str = Form(""),
        name: str = Form(""),
        instruction: str = Form("", description="默认语气指导，合成每一行时生效"),
        description: str = Form(""),
        language: str = Form(""),
        overwrite: bool = Form(False),
    ):
        if ref_audio is not None and ref_audio.filename:
            source: bytes | Path = await ref_audio.read()
        elif ref_audio_path:
            if (request.client.host if request.client else "") not in ("127.0.0.1", "::1", "localhost", "testclient"):
                raise HTTPException(403, "ref_audio_path 只允许本机调用，远程请上传 ref_audio 文件")
            source = Path(ref_audio_path)
            if not source.is_file():
                raise HTTPException(400, f"文件不存在：{ref_audio_path}")
        else:
            raise HTTPException(400, "需要 ref_audio（上传文件）或 ref_audio_path")
        vid = voice_id.strip() or slugify(name or Path(getattr(ref_audio, "filename", "") or ref_audio_path).stem)
        try:
            voice = await asyncio.to_thread(voices.create_clone, vid, name, source, ref_text,
                                            instruction=instruction, description=description,
                                            language=language, overwrite=overwrite)
        except VoiceError as exc:
            raise HTTPException(400, str(exc)) from exc
        manager.invalidate_voice(vid)
        return public_view(voice)

    @app.post("/v1/voices/design")
    async def create_design_voice(body: DesignVoiceRequest, request: Request):
        vid = (body.voice_id or "").strip() or slugify(body.name or "designed")
        if not body.freeze:
            try:
                voice = voices.create_design(vid, body.name, body.instruction, description=body.description,
                                             language=body.language, overwrite=body.overwrite)
            except VoiceError as exc:
                raise HTTPException(400, str(exc)) from exc
            return JSONResponse(public_view(voice), status_code=201)
        if voices.get(vid) is not None and not body.overwrite:
            raise HTTPException(400, f"音色已存在：{vid}（传 overwrite=true 覆盖）")
        require_ready()
        check_voice(None, body.engine)
        sample = body.sample_text or _default_sample(body.language, body.instruction)
        job = manager.submit("design_voice", body.model_dump() | {"voice_id": vid, "sample_text": sample})
        resp = await finish_or_accept(job, True, 300, request)
        if isinstance(resp, dict):
            return JSONResponse(public_view(voices.require(vid)) | {"job_id": job.id}, status_code=201)
        return resp

    @app.delete("/v1/voices/{voice_id}")
    def delete_voice(voice_id: str):
        try:
            voices.delete(voice_id)
        except VoiceError as exc:
            raise HTTPException(400 if voices.get(voice_id) else 404, str(exc)) from exc
        manager.invalidate_voice(voice_id)
        return {"deleted": voice_id}

    # -- synthesis --------------------------------------------------------------------------
    @app.post("/v1/tts")
    async def tts(body: TTSRequest, request: Request):
        require_ready()
        check_voice(body.voice, body.engine)
        job = manager.submit("tts", body.model_dump(exclude={"wait", "wait_seconds"}))
        return await finish_or_accept(job, body.wait, body.wait_seconds, request)

    @app.post("/v1/dub")
    async def dub(body: DubRequest, request: Request):
        require_ready()
        for i, line in enumerate(body.lines):
            check_voice(line.voice, line.engine or body.engine, f"第 {i + 1} 行：")
        payload = body.model_dump(exclude={"wait", "wait_seconds"})
        payload["lines"] = [{k: v for k, v in line.items() if v is not None} for line in payload["lines"]]
        job = manager.submit("dub", payload)
        return await finish_or_accept(job, body.wait, body.wait_seconds, request)

    @app.post("/v1/batch")
    async def batch(body: BatchRequest, request: Request):
        """批量合成互相独立的语音条目（每条一个文件，不拼接）。一次提交，服务按显存自动分批。"""
        require_ready()
        for i, item in enumerate(body.items):
            check_voice(item.voice, item.engine or body.engine, f"第 {i + 1} 条：")
        payload = body.model_dump(exclude={"wait", "wait_seconds"})
        payload["items"] = [{k: v for k, v in item.items() if v is not None} for item in payload["items"]]
        job = manager.submit("batch", payload)
        return await finish_or_accept(job, body.wait, body.wait_seconds, request)

    @app.post("/v1/audio/speech")
    async def openai_speech(body: OpenAISpeech):
        """OpenAI ``audio.speech`` compatible: returns the audio bytes."""
        require_ready()
        voice = None if body.voice in OPENAI_VOICES else body.voice
        engine = "qwen3" if "qwen" in body.model.lower() else ("breeze" if "breeze" in body.model.lower() else None)
        check_voice(voice, engine)
        fmt = {"opus": "ogg", "aac": "mp3", "pcm": "wav"}.get(body.response_format, body.response_format)
        job = manager.submit("tts", {"text": body.input, "voice": voice, "instruction": body.instructions,
                                     "format": fmt, "engine": engine})
        await asyncio.to_thread(job.event.wait, 3600)
        if job.status != "done":
            raise HTTPException(500, job.error or f"任务状态：{job.status}")
        data = Path(job.result["path"]).read_bytes()
        if body.response_format == "pcm":  # raw s16le 24 kHz mono, like OpenAI
            wav, _ = audio.read_audio(data)
            data = (wav.clip(-1, 1) * 32767).astype("<i2").tobytes()
            return Response(data, media_type="audio/pcm", headers={"X-Sample-Rate": "24000"})
        return Response(data, media_type=audio.media_type(fmt))

    # -- jobs & files -----------------------------------------------------------------------
    @app.get("/v1/jobs")
    def list_jobs(request: Request, limit: int = 50):
        return {"jobs": [job_view(j, request) for j in manager.list(limit)]}

    @app.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str, request: Request, wait_seconds: float = 0):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, f"任务不存在：{job_id}")
        if wait_seconds > 0 and job.status not in TERMINAL:
            await asyncio.to_thread(job.event.wait, min(wait_seconds, 3600))
        return job_view(job, request)

    @app.delete("/v1/jobs/{job_id}")
    def cancel_job(job_id: str):
        job = manager.get(job_id)
        if job is None:
            raise HTTPException(404, f"任务不存在：{job_id}")
        return {"id": job_id, "cancel_requested": manager.cancel(job), "status": job.status}

    @app.get("/files/{file_path:path}")
    def get_file(file_path: str):
        root = settings.outputs_dir.resolve()
        target = (root / file_path).resolve()
        if root not in target.parents or not target.is_file():
            raise HTTPException(404, "文件不存在")
        fmt = target.suffix.lstrip(".")
        media = {"srt": "application/x-subrip", "json": "application/json"}.get(fmt, audio.media_type(fmt))
        return FileResponse(target, media_type=media, filename=target.name)

    return app


def _default_sample(language: str, instruction: str) -> str:
    english = language.lower().startswith("en") or (instruction.isascii() and not language)
    if english:
        return ("Hello, it's nice to meet you. Let me tell you a story about a small town, "
                "where every morning began with the smell of fresh bread.")
    return "你好，很高兴认识你。让我给你讲一个故事，故事发生在一个安静的小镇上，那里的每个清晨都很温暖。"
