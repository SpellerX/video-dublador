"""Speech to text with word-level timestamps (faster-whisper / CTranslate2)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from .config import MODELS_DIR, tool_env
from .schema import Segment, Transcript, Word
from .utils import LOG, Progress, banner, read_json, write_json

#: faster-whisper model aliases and their approximate on-disk size.
WHISPER_MODELS: Dict[str, str] = {
    "tiny": "~75 MB",
    "tiny.en": "~75 MB",
    "base": "~145 MB",
    "base.en": "~145 MB",
    "small": "~488 MB",
    "small.en": "~488 MB",
    "medium": "~1.5 GB",
    "medium.en": "~1.5 GB",
    "large-v1": "~3.1 GB",
    "large-v2": "~3.1 GB",
    "large-v3": "~3.1 GB",
    "distil-large-v3": "~1.5 GB",
}

_WHISPER_CACHE: Dict[str, Any] = {}


def whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401,PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


def load_whisper_model(model_size: str, *, device: str = "cpu",
                       compute_type: str = "int8", threads: int = 0):
    """Load (and cache) a CTranslate2 Whisper model."""
    # Defensive: the module is usable on its own, without the pipeline having
    # resolved the "auto" sentinels first.
    if model_size in ("auto", "", None) or compute_type in ("auto", "", None):
        try:
            from .hardware import recommend  # noqa: PLC0415

            rec = recommend()
            if model_size in ("auto", "", None):
                model_size = rec.whisper_model
            if compute_type in ("auto", "", None):
                compute_type = rec.whisper_compute_type
            if not threads:
                threads = rec.threads
        except Exception:  # noqa: BLE001
            model_size = model_size if model_size not in ("auto", "", None) else "small"
            compute_type = compute_type if compute_type not in ("auto", "", None) else "int8"

    key = f"{model_size}|{device}|{compute_type}"
    if key in _WHISPER_CACHE:
        return _WHISPER_CACHE[key]

    from faster_whisper import WhisperModel  # noqa: PLC0415

    download_root = MODELS_DIR / "whisper"
    download_root.mkdir(parents=True, exist_ok=True)

    approx = WHISPER_MODELS.get(model_size, "unknown size")
    LOG.info("loading Whisper '%s' (%s) on %s/%s", model_size, approx, device, compute_type)
    if not any(download_root.glob(f"*{model_size}*")):
        LOG.info("  first run: downloading model into %s", download_root)

    kwargs: Dict[str, Any] = {
        "device": device,
        "compute_type": compute_type,
        "download_root": str(download_root),
    }
    if threads:
        kwargs["cpu_threads"] = threads
    if device == "cpu":
        kwargs["num_workers"] = 1

    try:
        model = WhisperModel(model_size, **kwargs)
    except Exception as e:  # noqa: BLE001
        LOG.error("failed to load Whisper '%s': %s", model_size, e)
        raise

    _WHISPER_CACHE[key] = model
    return model


def transcribe(
    audio_path: Path,
    *,
    model_size: str = "small",
    language: Optional[str] = None,
    device: str = "cpu",
    compute_type: str = "int8",
    beam_size: int = 5,
    vad_filter: bool = True,
    word_timestamps: bool = True,
    threads: int = 0,
    condition_on_previous_text: bool = True,
) -> Transcript:
    """Transcribe ``audio_path`` (16 kHz mono wav recommended)."""
    model = load_whisper_model(model_size, device=device, compute_type=compute_type, threads=threads)

    LOG.info("transcribing %s (language=%s, beam=%d, vad=%s)",
             Path(audio_path).name, language or "auto", beam_size, vad_filter)

    seg_iter, info = model.transcribe(
        str(audio_path),
        language=language,
        beam_size=beam_size,
        vad_filter=vad_filter,
        word_timestamps=word_timestamps,
        condition_on_previous_text=condition_on_previous_text,
        vad_parameters={"min_silence_duration_ms": 350} if vad_filter else None,
    )

    detected = getattr(info, "language", None)
    prob = float(getattr(info, "language_probability", 0.0) or 0.0)
    dur = float(getattr(info, "duration", 0.0) or 0.0)

    segments = []
    for i, s in enumerate(seg_iter):
        words = [
            Word(text=w.word.strip(), start=float(w.start), end=float(w.end),
                 prob=float(getattr(w, "probability", 1.0) or 1.0))
            for w in (getattr(s, "words", None) or [])
        ]
        segments.append(
            Segment(
                id=i,
                start=float(s.start),
                end=float(s.end),
                text=(s.text or "").strip(),
                words=words,
                no_speech_prob=float(getattr(s, "no_speech_prob", 0.0) or 0.0),
                avg_logprob=float(getattr(s, "avg_logprob", 0.0) or 0.0),
            )
        )
        if (i + 1) % 10 == 0:
            LOG.info("  ...%d segments (%.0fs of audio)", i + 1, segments[-1].end)

    if not dur and segments:
        dur = segments[-1].end

    tr = Transcript(
        language=detected,
        language_probability=prob,
        duration=dur,
        segments=segments,
        model=model_size,
        extra={"vad_filter": vad_filter, "beam_size": beam_size,
               "compute_type": compute_type, "device": device},
    )
    LOG.info("transcription done: %s", tr.summary())
    return tr


def transcribe_cached(
    audio_path: Path,
    json_out: Path,
    *,
    cache=None,
    **kwargs: Any,
) -> Transcript:
    """Transcribe with on-disk caching keyed by (audio, model, language)."""
    from .utils import StageCache  # noqa: PLC0415

    if cache is not None and cache.hit("transcribe", inputs=[audio_path], outputs=[json_out]):
        data = read_json(json_out)
        if data:
            return Transcript.from_dict(data)

    tr = transcribe(audio_path, **kwargs)
    write_json(json_out, tr.to_dict())
    if cache is not None:
        cache.mark("transcribe", inputs=[audio_path], outputs=[json_out],
                   model=kwargs.get("model_size"), language=kwargs.get("language"))
    return tr


def split_long_segments(tr: Transcript, max_seconds: float = 12.0) -> Transcript:
    """Dubbing quality degrades on very long lines; split them at word bounds.

    A segment longer than ``max_seconds`` is broken into pieces at the closest
    word boundary, keeping every original word in exactly one piece.
    """
    out = []
    next_id = 0
    for seg in tr.segments:
        if seg.duration <= max_seconds or not seg.words:
            seg.id = next_id
            next_id += 1
            out.append(seg)
            continue

        chunk: list[Word] = []
        chunk_start = seg.start
        for w in seg.words:
            if chunk and (w.end - chunk_start) > max_seconds:
                out.append(_mk(next_id, chunk_start, chunk[-1].end, chunk, seg))
                next_id += 1
                chunk = []
                chunk_start = w.start
            chunk.append(w)
        if chunk:
            out.append(_mk(next_id, chunk_start, chunk[-1].end, chunk, seg))
            next_id += 1

    tr.segments = out
    return tr


def _mk(sid: int, start: float, end: float, words: list[Word], parent: Segment) -> Segment:
    text = " ".join(w.text for w in words).strip() or parent.text
    return Segment(
        id=sid, start=start, end=end, text=text, words=list(words),
        no_speech_prob=parent.no_speech_prob, avg_logprob=parent.avg_logprob,
    )
