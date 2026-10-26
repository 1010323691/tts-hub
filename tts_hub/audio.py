"""Audio I/O and post-processing: reference decoding, trimming, loudness, concat, encode, SRT."""
from __future__ import annotations

import io
import shutil
import subprocess
from pathlib import Path

import numpy as np

SAMPLE_RATE = 24000
FORMATS = {
    # name: (soundfile format, subtype, media type)
    "wav": ("WAV", "PCM_16", "audio/wav"),
    "mp3": ("MP3", None, "audio/mpeg"),
    "flac": ("FLAC", "PCM_16", "audio/flac"),
    "ogg": ("OGG", "VORBIS", "audio/ogg"),
}
TARGET_RMS_DB = -20.0
PEAK_LIMIT = 0.97


def read_audio(source: bytes | str | Path) -> tuple[np.ndarray, int]:
    """Decode audio (wav/flac/ogg/mp3 natively; anything else through ffmpeg) to mono float32."""
    import soundfile as sf

    try:
        data, sr = sf.read(io.BytesIO(source) if isinstance(source, bytes) else str(source),
                           dtype="float32", always_2d=True)
    except Exception as exc:  # noqa: BLE001 — e.g. m4a/aac: try ffmpeg
        data, sr = _ffmpeg_decode(source, exc)
        data = data.reshape(-1, 1)
    mono = data.mean(axis=1).astype(np.float32)
    if mono.size == 0:
        raise ValueError("音频为空")
    return mono, int(sr)


def _ffmpeg_decode(source, original: Exception) -> tuple[np.ndarray, int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise ValueError(f"无法解码音频（未找到 ffmpeg）：{original}") from original
    args = [ffmpeg, "-v", "error", "-i", "pipe:0" if isinstance(source, bytes) else str(source),
            "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"]
    proc = subprocess.run(args, input=source if isinstance(source, bytes) else None,
                          capture_output=True, timeout=120)
    if proc.returncode != 0 or not proc.stdout:
        raise ValueError(f"无法解码音频：{proc.stderr.decode(errors='replace').strip() or original}")
    return np.frombuffer(proc.stdout, dtype="<f4").copy(), SAMPLE_RATE


def trim_silence(wav: np.ndarray, sr: int = SAMPLE_RATE, threshold_db: float = -45.0,
                 keep_ms: int = 60) -> np.ndarray:
    """Cut leading/trailing silence, keeping ``keep_ms`` of margin on each side."""
    if wav.size == 0:
        return wav
    frame = max(1, sr // 100)  # 10 ms
    n = wav.size // frame
    if n == 0:
        return wav
    rms = np.sqrt(np.mean(wav[:n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    loud = np.nonzero(20 * np.log10(rms) > threshold_db)[0]
    if loud.size == 0:
        return wav
    keep = int(sr * keep_ms / 1000)
    start = max(0, loud[0] * frame - keep)
    end = min(wav.size, (loud[-1] + 1) * frame + keep)
    return wav[start:end]


def normalize_loudness(wav: np.ndarray, target_db: float = TARGET_RMS_DB) -> np.ndarray:
    """Scale speech to a common RMS level, then peak-limit (keeps lines of one script even)."""
    if wav.size == 0:
        return wav
    rms = float(np.sqrt(np.mean(wav ** 2)))
    if rms < 1e-5:
        return wav
    out = wav * (10 ** (target_db / 20) / rms)
    peak = float(np.max(np.abs(out)))
    if peak > PEAK_LIMIT:
        out = out * (PEAK_LIMIT / peak)
    return out.astype(np.float32)


def silence(ms: int, sr: int = SAMPLE_RATE) -> np.ndarray:
    return np.zeros(max(0, int(sr * ms / 1000)), dtype=np.float32)


def concat(segments: list[np.ndarray], gaps_ms: list[int], sr: int = SAMPLE_RATE
           ) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """Join ``segments`` with ``gaps_ms[i]`` of silence after segment i (last gap ignored).

    Returns the audio and each segment's (start_s, end_s) on the joined timeline.
    """
    parts: list[np.ndarray] = []
    spans: list[tuple[float, float]] = []
    cursor = 0
    for i, seg in enumerate(segments):
        if i:
            gap = silence(gaps_ms[i - 1], sr)
            parts.append(gap)
            cursor += gap.size
        parts.append(seg)
        spans.append((cursor / sr, (cursor + seg.size) / sr))
        cursor += seg.size
    audio = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return audio, spans


def encode(wav: np.ndarray, fmt: str, sr: int = SAMPLE_RATE) -> bytes:
    import soundfile as sf

    if fmt not in FORMATS:
        raise ValueError(f"不支持的格式：{fmt}（可选：{', '.join(FORMATS)}）")
    sf_format, subtype, _ = FORMATS[fmt]
    buf = io.BytesIO()
    kwargs = {"format": sf_format}
    if subtype:
        kwargs["subtype"] = subtype
    if fmt == "mp3":
        kwargs["compression_level"] = 0.3  # ~128 kbps VBR-ish; libsndfile's default is lower
    sf.write(buf, np.clip(wav, -1.0, 1.0), sr, **kwargs)
    return buf.getvalue()


def media_type(fmt: str) -> str:
    return FORMATS.get(fmt, (None, None, "application/octet-stream"))[2]


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(entries: list[dict]) -> str:
    """``entries``: dicts with start, end, text and optional speaker."""
    blocks = []
    for n, e in enumerate(entries, 1):
        label = f"{e['speaker']}：" if e.get("speaker") else ""
        blocks.append(f"{n}\n{_srt_time(e['start'])} --> {_srt_time(e['end'])}\n{label}{e['text']}\n")
    return "\n".join(blocks)
