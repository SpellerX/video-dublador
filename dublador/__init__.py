"""Video dubbing pipeline.

Transcript -> translate -> clone each actor's voice (F5-TTS) -> re-time the
lines onto the original clock -> re-animate lips (Wav2Lip) -> dubbed video.

The package is intentionally import-light: heavy dependencies (torch, F5-TTS,
faster-whisper) are imported lazily inside the stage that needs them, so
``dublador check`` and ``dublador languages`` start instantly and still work
when an optional component is missing.
"""
from __future__ import annotations

__version__ = "1.0.0"

from .config import DubladorConfig, LANGUAGES, normalize_lang
from .schema import Segment, SpeakerProfile, Transcript, Word

__all__ = [
    "__version__",
    "DubladorConfig",
    "LANGUAGES",
    "normalize_lang",
    "Segment",
    "SpeakerProfile",
    "Transcript",
    "Word",
    "run_pipeline",
    "main",
]


def __getattr__(name: str):
    """Lazily expose the heavier entry points without importing them eagerly."""
    if name == "run_pipeline":
        from .pipeline import run_pipeline  # noqa: PLC0415

        return run_pipeline
    if name == "main":
        from .cli import main  # noqa: PLC0415

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
