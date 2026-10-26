"""Job queue: one worker thread owns the GPU and runs jobs FIFO.

Every synthesis request becomes a job (``tts``, ``dub``, ``batch`` or ``design_voice``); the HTTP layer
waits on it for synchronous calls, or hands the caller a job id to poll. Finished job records
are written to ``data/jobs/<id>.json`` so results survive a restart; outputs live in
``data/outputs/<id>/``.

Several engines (``breeze``, ``qwen3``) share the queue. The default engine loads at startup;
the others load the first time a request names them. Each line of a script picks its engine
(the request's, the line's own, or the one its voice requires), so one script may mix engines.
"""
from __future__ import annotations

import json
import logging
import queue
import threading
import time
import traceback
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio
from .config import Settings
from .engine import Row, RowResult
from .text import split_text, text_chars
from .voices import BUILTIN_VOICES, VoiceError, VoiceStore, voice_engines

log = logging.getLogger("tts_hub.jobs")

DEFAULT_VOICE = "zh_female_narrator"
TERMINAL = ("done", "error", "cancelled")
# Qwen3-TTS language names, keyed by the short codes the API also accepts
LANGUAGES = {"zh": "chinese", "en": "english", "ja": "japanese", "ko": "korean", "de": "german",
             "fr": "french", "es": "spanish", "it": "italian", "pt": "portuguese", "ru": "russian"}


def qwen_language(value: str | None) -> str:
    value = (value or "").strip().lower()
    return LANGUAGES.get(value, value) if value not in ("", "auto") else ""


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    type: str
    request: dict
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    progress: dict = field(default_factory=lambda: {"done": 0, "total": 0})
    result: dict | None = None
    error: str | None = None
    cancel_requested: bool = False
    event: threading.Event = field(default_factory=threading.Event, repr=False)

    def view(self) -> dict:
        return {"id": self.id, "type": self.type, "status": self.status,
                "created_at": self.created_at, "started_at": self.started_at,
                "finished_at": self.finished_at, "progress": self.progress,
                "result": self.result, "error": self.error}


class JobManager:
    def __init__(self, settings: Settings, voices: VoiceStore, engines: dict):
        self.settings = settings
        self.voices = voices
        self.engines = engines
        self.default_engine = settings.default_engine if settings.default_engine in engines else next(iter(engines))
        self.engine_errors: dict[str, str] = {}
        self.jobs: dict[str, Job] = {}
        self._queue: "queue.Queue[Job | str | None]" = queue.Queue()
        self._lock = threading.Lock()
        # (engine, voice id) -> (updated_at, prompt): prompt formats differ per engine
        self._prompts: dict[tuple[str, str], tuple[float, object]] = {}
        # starting | loading | seeding | warming | ready | unloaded | stopping | error
        self.state = "starting"
        self.state_error: str | None = None
        self.last_activity = time.time()
        self.unloaded_at: float | None = None
        # Called (from the worker thread) to stop the HTTP server after the idle shutdown.
        self.on_shutdown = None
        self._thread = threading.Thread(target=self._run, name="breeze-gpu", daemon=True)
        self._load_history()

    # -- lifecycle --------------------------------------------------------------------------
    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._queue.put(None)

    def _load_history(self) -> None:
        for path in sorted(self.settings.jobs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                job = Job(id=data["id"], type=data["type"], request=data.get("request", {}),
                          status=data["status"], created_at=data.get("created_at", 0),
                          started_at=data.get("started_at"), finished_at=data.get("finished_at"),
                          progress=data.get("progress", {}), result=data.get("result"),
                          error=data.get("error"))
                job.event.set()
                self.jobs[job.id] = job
            except (OSError, ValueError, KeyError):
                continue

    def _persist(self, job: Job) -> None:
        data = job.view() | {"request": job.request}
        path = self.settings.jobs_dir / f"{job.id}.json"
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    # -- public API -------------------------------------------------------------------------
    def submit(self, job_type: str, request: dict) -> Job:
        job = Job(id=f"{job_type}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}",
                  type=job_type, request=request)
        with self._lock:
            self.jobs[job.id] = job
        self._queue.put(job)
        return job

    def get(self, job_id: str) -> Job | None:
        return self.jobs.get(job_id)

    def list(self, limit: int = 50) -> list[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_at, reverse=True)[:limit]

    def queue_position(self, job: Job) -> int:
        queued = [j for j in self.jobs.values() if j.status == "queued"]
        queued.sort(key=lambda j: j.created_at)
        return next((i for i, j in enumerate(queued) if j.id == job.id), -1)

    def cancel(self, job: Job) -> bool:
        if job.status in TERMINAL:
            return False
        job.cancel_requested = True
        if job.status == "queued":
            self._finish(job, "cancelled", error="已取消")
        return True

    def touch(self) -> None:
        """Record API activity; it postpones the idle unload and shutdown."""
        self.last_activity = time.time()

    def request(self, command: str) -> None:
        """Queue ``unload`` or ``shutdown`` for the worker (it alone touches the engines)."""
        self._queue.put(command)

    def idle_view(self) -> dict:
        s = self.settings
        idle = time.time() - self.last_activity
        view = {"idle_s": round(idle), "unload_after_s": s.idle_unload_s or None,
                "shutdown_after_s": s.idle_shutdown_s or None}
        if self.state == "ready" and s.idle_unload_s:
            view["unload_in_s"] = max(0, round(s.idle_unload_s - idle))
        deadline = self._shutdown_deadline()
        if deadline is not None:
            view["shutdown_in_s"] = max(0, round(deadline - time.time()))
        return view

    def engines_view(self) -> dict:
        view = {}
        for name, engine in self.engines.items():
            entry = {"loaded": bool(engine.loaded), "default": name == self.default_engine,
                     "modes": list(getattr(engine, "modes", ()))}
            if hasattr(engine, "loaded_models"):
                entry["models"] = engine.loaded_models()
            if getattr(engine, "last_plan", None):
                entry["last_batches"] = engine.last_plan  # rows per generate call of the last job
            if getattr(engine, "oom_scale", 1.0) < 1.0:
                entry["oom_scale"] = round(engine.oom_scale, 3)
            if name in self.engine_errors:
                entry["error"] = self.engine_errors[name]
            view[name] = entry
        return view

    def vram(self) -> dict | None:
        for engine in self.engines.values():
            stats = engine.vram() if hasattr(engine, "vram") else None
            if stats:
                return stats
        return None

    def resolve_engine(self, requested: str | None, voice: dict | None) -> str:
        """The engine a line runs on: the one named, else the voice's only engine, else default."""
        name = (requested or "").strip().lower() or None
        if name is not None and name not in self.engines:
            raise VoiceError(f"未知引擎：{name}（可选：{', '.join(self.engines)}）")
        if voice is not None:
            allowed = voice_engines(voice)
            if name is None:
                name = self.default_engine if self.default_engine in allowed else allowed[0]
            elif name not in allowed:
                raise VoiceError(f"音色 {voice['id']} 只能用于 {'/'.join(allowed)} 引擎，不能用 {name}")
        return name or self.default_engine

    def invalidate_voice(self, voice_id: str) -> None:
        for key in [k for k in self._prompts if k[1] == voice_id]:
            self._prompts.pop(key, None)

    def stats(self) -> dict:
        counts: dict[str, int] = {}
        for job in self.jobs.values():
            counts[job.status] = counts.get(job.status, 0) + 1
        return counts

    # -- worker -----------------------------------------------------------------------------
    def _run(self) -> None:
        try:
            self.state = "loading"
            self._ensure(self.default_engine)
            if self.settings.seed_builtin_voices:
                self.state = "seeding"
                self._seed_builtin_voices()
            self.state = "warming"
            self._warmup()
            self.state = "ready"
            log.info("服务就绪（默认引擎 %s）", self.default_engine)
        except Exception as exc:  # noqa: BLE001
            self.state, self.state_error = "error", f"{type(exc).__name__}: {exc}"
            log.exception("默认引擎加载失败")
        self.touch()
        while True:
            try:
                job = self._queue.get(timeout=self.settings.idle_check_s)
            except queue.Empty:
                if self._idle_tick():
                    return
                continue
            if job is None:
                return
            if job == "unload":
                self._unload()
                continue
            if job == "shutdown":
                self._shutdown("收到关闭请求")
                return
            if job.status != "queued":
                continue
            job.status, job.started_at = "running", time.time()
            try:
                handler = {"tts": self._run_tts, "dub": self._run_dub, "batch": self._run_batch,
                           "design_voice": self._run_design_voice}[job.type]
                result = handler(job)
                self._finish(job, "done", result=result)
            except JobCancelled:
                self._finish(job, "cancelled", error="已取消")
            except (VoiceError, ValueError) as exc:
                self._finish(job, "error", error=str(exc))
            except Exception as exc:  # noqa: BLE001
                log.error("任务 %s 失败：%s", job.id, traceback.format_exc())
                self._finish(job, "error", error=f"{type(exc).__name__}: {exc}")
                if "out of memory" in str(exc).lower():
                    import torch
                    torch.cuda.empty_cache()

    def _ensure(self, name: str):
        """The loaded engine ``name``, loading it now if needed (worker thread only)."""
        engine = self.engines[name]
        if engine.loaded:
            return engine
        previous = self.state
        self.state = "loading"
        try:
            t = time.time()
            engine.load()
            self.engine_errors.pop(name, None)
            log.info("已加载引擎 %s（%.1fs）", name, time.time() - t)
        except Exception as exc:
            self.engine_errors[name] = f"{type(exc).__name__}: {exc}"
            self.state = previous if previous != "loading" else "error"
            raise RuntimeError(f"{name} 引擎加载失败：{exc}") from exc
        self.state, self.unloaded_at, self.state_error = "ready", None, None
        return engine

    # -- idle lifecycle (worker thread) -----------------------------------------------------
    def _shutdown_deadline(self) -> float | None:
        """Unload and shutdown are consecutive idle phases: shutdown comes ``idle_shutdown_s``
        after the unload (or after the last request when unloading is disabled)."""
        s = self.settings
        if not s.idle_shutdown_s or self.state not in ("ready", "unloaded"):
            return None
        return self.last_activity + (s.idle_unload_s or 0) + s.idle_shutdown_s

    def _idle_tick(self) -> bool:
        """Runs whenever the queue stays empty for ``idle_check_s``; True = worker must exit."""
        s, now = self.settings, time.time()
        if self.state == "ready" and s.idle_unload_s and now - self.last_activity >= s.idle_unload_s:
            log.info("空闲 %.0f 分钟，卸载模型权重", (now - self.last_activity) / 60)
            self._unload()
        deadline = self._shutdown_deadline()
        if deadline is not None and now >= deadline:
            self._shutdown(f"空闲 {(now - self.last_activity) / 60:.0f} 分钟")
            return True
        return False

    def _unload(self) -> None:
        if self.state != "ready":
            return
        for engine in self.engines.values():
            if engine.loaded:
                engine.unload()
        self.state, self.unloaded_at = "unloaded", time.time()

    def _shutdown(self, reason: str) -> None:
        log.info("自动关闭服务：%s", reason)
        self.state = "stopping"
        for engine in self.engines.values():
            if engine.loaded:
                engine.unload()
        if self.on_shutdown is not None:
            self.on_shutdown()
        else:
            log.warning("未注册 on_shutdown（非 python -m tts_hub 启动），进程不会自行退出")

    def _finish(self, job: Job, status: str, result=None, error=None) -> None:
        job.status, job.result, job.error = status, result, error
        job.finished_at = time.time()
        self.touch()
        try:
            self._persist(job)
        except OSError:
            log.exception("无法写入任务记录 %s", job.id)
        job.event.set()

    def _progress(self, job: Job):
        def report(done: int, total: int) -> None:
            job.progress = {"done": done, "total": total}
            if job.cancel_requested:
                raise JobCancelled()
        return report

    # -- voices -----------------------------------------------------------------------------
    def _prompt(self, voice: dict, engine_name: str):
        key = (engine_name, voice["id"])
        cached = self._prompts.get(key)
        if cached and cached[0] == voice.get("updated_at"):
            return cached[1]
        wav, sr = self.voices.load_reference(voice)
        prompt = self._ensure(engine_name).encode_reference(wav, sr, voice["ref_text"])
        self._prompts[key] = (voice.get("updated_at"), prompt)
        return prompt

    def _resolve_voice(self, voice_id: str | None) -> dict | None:
        voice_id = (voice_id or "").strip()
        if not voice_id:
            return self.voices.get(DEFAULT_VOICE)
        return self.voices.require(voice_id)

    def _row(self, text: str, voice: dict | None, instruction: str, cfg_scale, seed: int,
             engine: str, language: str = "", notes: set | None = None) -> Row:
        """A row for ``text`` spoken by ``voice`` on ``engine`` with an optional per-line direction."""
        instruction = (instruction or "").strip()
        # only an explicit language: a voice's own language would force e.g. English lines of a
        # Chinese voice to be read as Chinese; Qwen3's "auto" detects per line
        lang = qwen_language(language)
        if voice is None:
            return Row(text, None, instruction, cfg_scale, seed, language=lang)
        if voice["kind"] == "preset":
            return Row(text, None, instruction or voice.get("instruction", ""), cfg_scale, seed,
                       speaker=voice["speaker"], language=lang)
        if voice["kind"] == "clone":
            direction = instruction or voice.get("instruction", "")
            if direction and engine == "qwen3" and notes is not None:
                notes.add("Qwen3 克隆音色不支持 instruction（语气指导已忽略）；需要控制语气请用预置音色或 breeze 引擎")
            return Row(text, self._prompt(voice, engine), direction, cfg_scale, seed, language=lang)
        combined = " ".join(p for p in (voice.get("instruction", ""), instruction) if p)
        return Row(text, None, combined, cfg_scale, seed, language=lang)

    def _synthesize(self, job: Job | None, rows: list[Row], engines: list[str]) -> list[RowResult]:
        """Run ``rows[i]`` on ``engines[i]``; results in input order."""
        results: list[RowResult | None] = [None] * len(rows)
        groups: dict[str, list[int]] = {}
        for i, name in enumerate(engines):
            groups.setdefault(name, []).append(i)
        report = self._progress(job) if job else None
        done = 0
        for name, positions in groups.items():
            engine = self._ensure(name)
            offset = done

            def progress(d, _t, offset=offset):
                if report:
                    report(offset + d, len(rows))

            for pos, res in zip(positions, engine.synthesize([rows[p] for p in positions], progress)):
                results[pos] = res
            done += len(positions)
        return results  # type: ignore[return-value]

    def _seed_builtin_voices(self) -> None:
        missing = [spec for spec in BUILTIN_VOICES if self.voices.get(spec["id"]) is None]
        if not missing:
            return
        log.info("生成内置音色：%s", ", ".join(s["id"] for s in missing))
        rows = [Row(s["sample"], None, s["instruction"], None, 20261026, language=qwen_language(s["language"]))
                for s in missing]
        results = self._synthesize(None, rows, [self.default_engine] * len(rows))
        for spec, res in zip(missing, results):
            wav = audio.normalize_loudness(audio.trim_silence(res.audio))
            self.voices.create_clone(
                spec["id"], spec["name"], audio.encode(wav, "wav"), spec["sample"],
                language=spec["language"], description=spec["instruction"], overwrite=True,
                extra={"builtin": True, "source": "design", "design_instruction": spec["instruction"],
                       "design_engine": self.default_engine})

    def _warmup(self) -> None:
        """One short clone row, so the first real request does not pay kernel/graph warmup."""
        voice = self.voices.get(DEFAULT_VOICE)
        try:
            t = time.time()
            name = self.resolve_engine(None, voice)
            self._synthesize(None, [self._row("你好，欢迎使用。", voice, "", None, 1, name)], [name])
            log.info("预热完成（%.1fs）", time.time() - t)
        except Exception:  # noqa: BLE001 — warmup is best effort
            log.exception("预热失败（不影响服务）")

    def _run_design_voice(self, job: Job) -> dict:
        r = job.request
        name = self.resolve_engine(r.get("engine"), None)
        rows = [Row(r["sample_text"], None, r["instruction"], None, int(r.get("seed", 42)),
                    language=qwen_language(r.get("language")))]
        res = self._synthesize(job, rows, [name])[0]
        wav = audio.normalize_loudness(audio.trim_silence(res.audio))
        voice = self.voices.create_clone(
            r["voice_id"], r.get("name") or r["voice_id"], audio.encode(wav, "wav"), r["sample_text"],
            description=r.get("description", ""), language=r.get("language", ""),
            overwrite=bool(r.get("overwrite")),
            extra={"source": "design", "design_instruction": r["instruction"], "design_engine": name})
        self.invalidate_voice(voice["id"])
        return {"voice": voice["id"], "ref_seconds": voice["ref_seconds"], "engine": name}

    # -- synthesis --------------------------------------------------------------------------
    def _post(self, wav: np.ndarray, request: dict) -> np.ndarray:
        if request.get("trim_silence", True):
            wav = audio.trim_silence(wav)
        if request.get("normalize", True):
            wav = audio.normalize_loudness(wav)
        return wav

    def _out_dir(self, job: Job) -> Path:
        path = self.settings.outputs_dir / job.id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _file_entry(self, job: Job, path: Path) -> dict:
        rel = path.relative_to(self.settings.outputs_dir).as_posix()
        return {"path": str(path.resolve()), "url": f"/files/{rel}"}

    def _run_tts(self, job: Job) -> dict:
        r = job.request
        voice = self._resolve_voice(r.get("voice"))
        name = self.resolve_engine(r.get("engine"), voice)
        chunks = split_text(r["text"], self.settings.max_chunk_chars) if r.get("split", True) else [r["text"].strip()]
        if not chunks:
            raise ValueError("text 为空")
        seed = int(r.get("seed", 42))
        notes: set = set()
        rows = [self._row(c, voice, r.get("instruction", ""), r.get("cfg_scale"), seed, name,
                          r.get("language", ""), notes) for c in chunks]
        results = self._synthesize(job, rows, [name] * len(rows))
        wavs = [self._post(res.audio, r) for res in results]
        gap = int(r.get("chunk_pause_ms", 200))
        full, spans = audio.concat(wavs, [gap] * len(wavs))
        fmt = r.get("format", "wav")
        out = self._out_dir(job) / f"speech.{fmt}"
        out.write_bytes(audio.encode(full, fmt))
        return {
            **self._file_entry(job, out), "format": fmt, "sample_rate": audio.SAMPLE_RATE,
            "duration_s": round(full.size / audio.SAMPLE_RATE, 3),
            "voice": voice["id"] if voice else None, "engine": name,
            "chunks": [{"text": c, "start": round(s, 3), "end": round(e, 3), "capped": res.capped}
                       for c, (s, e), res in zip(chunks, spans, results)],
            "warnings": sorted(notes) + _warnings(chunks, results),
        }

    def _synthesize_lines(self, job: Job, lines: list[dict], r: dict, label: str = "行"):
        """Every line (split into chunks) synthesized in one pass across engines.

        Returns per-line ``(wavs, capped, engines, voices, notes)``; ``r`` carries the request-level
        defaults (engine, language, seed, cfg_scale, post-processing).
        """
        seed = int(r.get("seed", 42))
        voices: dict[str, dict | None] = {}
        line_engines: list[str] = []
        rows, owners, row_engines = [], [], []  # owners[i] = line index of row i
        notes: set = set()
        for li, line in enumerate(lines):
            key = line.get("voice") or ""
            if key not in voices:
                voices[key] = self._resolve_voice(key)
            try:
                name = self.resolve_engine(line.get("engine") or r.get("engine"), voices[key])
            except VoiceError as exc:
                raise VoiceError(f"第 {li + 1} {label}：{exc}") from exc
            line_engines.append(name)
            chunks = split_text(line["text"], self.settings.max_chunk_chars)
            if not chunks:
                raise ValueError(f"第 {li + 1} {label} text 为空")
            for c in chunks:
                rows.append(self._row(c, voices[key], line.get("instruction", ""),
                                      line.get("cfg_scale", r.get("cfg_scale")),
                                      int(line.get("seed", seed)), name,
                                      line.get("language") or r.get("language", ""), notes))
                owners.append(li)
                row_engines.append(name)
        results = self._synthesize(job, rows, row_engines)

        # stitch chunks back into lines
        line_audio: list[list[np.ndarray]] = [[] for _ in lines]
        line_capped = [False] * len(lines)
        for owner, res in zip(owners, results):
            line_audio[owner].append(self._post(res.audio, r))
            line_capped[owner] |= res.capped
        chunk_gap = int(r.get("chunk_pause_ms", 200))
        line_wavs = [audio.concat(parts, [chunk_gap] * len(parts))[0] for parts in line_audio]
        return line_wavs, line_capped, line_engines, voices, notes

    def _run_batch(self, job: Job) -> dict:
        """Independent utterances: one file per item, no merge, no subtitles."""
        r = job.request
        items = r["items"]
        fmt = r.get("format", "wav")
        wavs, capped, engines, voices, notes = self._synthesize_lines(job, items, r, "条")
        out_dir = self._out_dir(job)
        manifest = []
        for i, (item, wav) in enumerate(zip(items, wavs)):
            voice = voices[item.get("voice") or ""]
            path = out_dir / f"{item['id']}.{fmt}"
            path.write_bytes(audio.encode(wav, fmt))
            manifest.append({"index": i, "id": item["id"], "text": item["text"],
                             "voice": voice["id"] if voice else None, "engine": engines[i],
                             "duration_s": round(wav.size / audio.SAMPLE_RATE, 3), "capped": capped[i],
                             "file": self._file_entry(job, path)})
        mf = out_dir / "manifest.json"
        mf.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "output_dir": str(out_dir.resolve()), "format": fmt, "sample_rate": audio.SAMPLE_RATE,
            "count": len(items), "engines": sorted(set(engines)),
            "total_duration_s": round(sum(m["duration_s"] for m in manifest), 3),
            "manifest": self._file_entry(job, mf), "items": manifest,
            "warnings": sorted(notes) + [f"第 {m['index'] + 1} 条（{m['id']}）疑似解码失控（已重试），请试听"
                                         for m in manifest if m["capped"]],
        }

    def _run_dub(self, job: Job) -> dict:
        r = job.request
        lines = r["lines"]
        fmt = r.get("format", "wav")
        line_wavs, line_capped, line_engines, voices, notes = self._synthesize_lines(job, lines, r)

        pause = int(r.get("pause_ms", 400))
        same = int(r.get("same_speaker_pause_ms", 250))
        gaps = []
        for li, line in enumerate(lines):
            if line.get("pause_after_ms") is not None:
                gaps.append(max(0, int(line["pause_after_ms"])))
            elif li + 1 < len(lines):
                gaps.append(same if lines[li + 1].get("voice") == line.get("voice") else pause)
            else:
                gaps.append(0)
        full, spans = audio.concat(line_wavs, gaps)
        lead = int(r.get("lead_silence_ms", 0))
        if lead:
            full = np.concatenate([audio.silence(lead), full])
            spans = [(s + lead / 1000, e + lead / 1000) for s, e in spans]

        out_dir = self._out_dir(job)
        merged = out_dir / f"{r.get('name') or 'dub'}.{fmt}"
        merged.write_bytes(audio.encode(full, fmt))
        timeline = []
        width = max(3, len(str(len(lines))))
        save_lines = r.get("save_lines", True)
        if save_lines:
            (out_dir / "lines").mkdir(exist_ok=True)
        for li, (line, wav, (start, end)) in enumerate(zip(lines, line_wavs, spans)):
            voice = voices[line.get("voice") or ""]
            entry = {"index": li, "voice": voice["id"] if voice else None, "engine": line_engines[li],
                     "speaker": line.get("speaker") or (voice or {}).get("name") or "",
                     "text": line["text"], "start": round(start, 3), "end": round(end, 3),
                     "duration_s": round(wav.size / audio.SAMPLE_RATE, 3), "capped": line_capped[li]}
            if save_lines:
                path = out_dir / "lines" / f"{li + 1:0{width}d}_{entry['voice'] or 'design'}.{fmt}"
                path.write_bytes(audio.encode(wav, fmt))
                entry["file"] = self._file_entry(job, path)
            timeline.append(entry)
        srt_entries = [{"start": t["start"], "end": t["end"], "text": t["text"],
                        "speaker": t["speaker"] if r.get("srt_speaker", False) else ""} for t in timeline]
        srt = out_dir / f"{r.get('name') or 'dub'}.srt"
        srt.write_text(audio.to_srt(srt_entries), encoding="utf-8")
        tl = out_dir / "timeline.json"
        tl.write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            **self._file_entry(job, merged), "format": fmt, "sample_rate": audio.SAMPLE_RATE,
            "duration_s": round(full.size / audio.SAMPLE_RATE, 3),
            "engines": sorted(set(line_engines)),
            "srt": self._file_entry(job, srt), "timeline_json": self._file_entry(job, tl),
            "output_dir": str(out_dir.resolve()), "lines": timeline,
            "warnings": sorted(notes) + [f"第 {t['index'] + 1} 行疑似解码失控（已重试），请试听"
                                         for t in timeline if t["capped"]],
        }


def _warnings(chunks: list[str], results) -> list[str]:
    return [f"片段「{c[:20]}」疑似解码失控（已重试），请试听" for c, res in zip(chunks, results) if res.capped]


def estimate_chars(request: dict) -> int:
    if "lines" in request:
        return sum(text_chars(line.get("text", "")) for line in request["lines"])
    return text_chars(request.get("text", ""))
