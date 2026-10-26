"""Text chunking and decode caps (pure, stdlib-only)."""
from __future__ import annotations

import math
import re

CODEC_FRAME_RATE = 12.5   # Breeze codec: 12.5 frames per second of 24 kHz audio
MAX_NEW_TOKENS = 1500     # upstream server limit (120 s of audio)
# Length-proportional decode cap. Natural Chinese speech is ~2.8 frames/char; 7/char leaves
# room for slow instructions and vocal events while a runaway row still stops early (a batch
# decodes until its longest row's cap, so one runaway row would otherwise stall the batch).
FRAMES_PER_CHAR = 7
FRAME_CAP_FLOOR = 150     # 12 s: ultra-short lines and long inline events

_SENTENCE_END = re.compile(r"(?<=[。！？!?；;…\n])|(?<=[.](?=\s))")
_CLAUSE_END = re.compile(r"(?<=[，,、：:])")


def text_chars(text: str) -> int:
    """Characters that drive the decode length (whitespace ignored)."""
    return len(re.sub(r"\s+", "", text or ""))


def frame_cap(chars: int) -> int:
    return min(MAX_NEW_TOKENS, max(FRAME_CAP_FLOOR, int(math.ceil(chars * FRAMES_PER_CHAR))))


def _pieces(text: str, pattern: re.Pattern) -> list[str]:
    return [p for p in (s.strip() for s in pattern.split(text)) if p]


def _hard_split(text: str, max_chars: int) -> list[str]:
    return [text[i:i + max_chars] for i in range(0, len(text), max_chars)]


def _pack(pieces: list[str], max_chars: int, joiner_for) -> list[str]:
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not current:
            current = piece
        elif text_chars(current) + text_chars(piece) <= max_chars:
            current = current + joiner_for(current) + piece
        else:
            chunks.append(current)
            current = piece
    if current:
        chunks.append(current)
    return chunks


def _joiner(previous: str) -> str:
    # Latin text keeps its word gap; CJK sentences join without one.
    return " " if previous and previous[-1].isascii() and previous[-1] not in "\n" else ""


def split_text(text: str, max_chars: int = 120) -> list[str]:
    """Split ``text`` into synthesis chunks of at most ``max_chars`` (sentence-first).

    Sentences are packed greedily; an over-long sentence is split at clause punctuation,
    and only an over-long clause is cut mid-text.
    """
    text = (text or "").strip()
    if not text:
        return []
    if text_chars(text) <= max_chars:
        return [text]
    pieces: list[str] = []
    for sentence in _pieces(text, _SENTENCE_END):
        if text_chars(sentence) <= max_chars:
            pieces.append(sentence)
            continue
        for clause in _pieces(sentence, _CLAUSE_END):
            pieces.extend([clause] if text_chars(clause) <= max_chars else _hard_split(clause, max_chars))
    return _pack(pieces, max_chars, _joiner)
