"""Serialisable data model shared by every pipeline stage."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Word:
    text: str
    start: float
    end: float
    prob: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Word":
        return cls(text=d["text"], start=float(d["start"]), end=float(d["end"]),
                   prob=float(d.get("prob", 1.0)))


@dataclass
class Segment:
    """One utterance. Carries every intermediate artefact for the stage cache."""

    id: int
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)

    speaker: Optional[str] = None
    speaker_confidence: float = 0.0
    translation: Optional[str] = None

    #: synthesised speech for this line (path to a wav)
    tts_audio: Optional[str] = None
    #: the same clip, time-fitted to the original slot
    fitted_audio: Optional[str] = None
    fit_meta: Dict[str, Any] = field(default_factory=dict)

    no_speech_prob: float = 0.0
    avg_logprob: float = 0.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    @property
    def text_for_tts(self) -> str:
        return (self.translation or self.text or "").strip()

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Segment":
        d = dict(d)
        d["words"] = [Word.from_dict(w) for w in d.get("words", [])]
        return cls(**d)


@dataclass
class SpeakerProfile:
    """A detected voice and the reference clip used to clone it with F5-TTS."""

    id: str
    segments: List[int] = field(default_factory=list)
    total_speech: float = 0.0
    ref_audio: Optional[str] = None
    ref_text: Optional[str] = None
    ref_start: float = 0.0
    ref_end: float = 0.0
    embedding: Optional[List[float]] = None
    gender: Optional[str] = None
    mean_pitch_hz: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SpeakerProfile":
        return cls(**d)


@dataclass
class Transcript:
    language: Optional[str] = None
    language_probability: float = 0.0
    duration: float = 0.0
    segments: List[Segment] = field(default_factory=list)
    speakers: Dict[str, SpeakerProfile] = field(default_factory=dict)
    model: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "language": self.language,
            "language_probability": self.language_probability,
            "duration": self.duration,
            "model": self.model,
            "extra": self.extra,
            "segments": [s.to_dict() for s in self.segments],
            "speakers": {k: v.to_dict() for k, v in self.speakers.items()},
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Transcript":
        return cls(
            language=d.get("language"),
            language_probability=float(d.get("language_probability") or 0.0),
            duration=float(d.get("duration") or 0.0),
            model=d.get("model", ""),
            extra=d.get("extra", {}) or {},
            segments=[Segment.from_dict(s) for s in d.get("segments", [])],
            speakers={k: SpeakerProfile.from_dict(v) for k, v in (d.get("speakers") or {}).items()},
        )

    # -- convenience --------------------------------------------------------
    @property
    def speech_time(self) -> float:
        return sum(s.duration for s in self.segments)

    def speaker_ids(self) -> List[str]:
        return sorted({s.speaker for s in self.segments if s.speaker})

    def to_srt(self, *, translated: bool = True) -> str:
        def ts(t: float) -> str:
            t = max(0.0, t)
            h, rem = divmod(int(t), 3600)
            m, s = divmod(rem, 60)
            ms = int(round((t - int(t)) * 1000))
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        out: List[str] = []
        for i, seg in enumerate(self.segments, 1):
            text = seg.translation if (translated and seg.translation) else seg.text
            tag = f" [{seg.speaker}]" if seg.speaker else ""
            out.append(f"{i}\n{ts(seg.start)} --> {ts(seg.end)}\n{tag}{text}\n")
        return "\n".join(out)

    def summary(self) -> str:
        langs = f"{self.language} ({self.language_probability:.0%})" if self.language else "?"
        return (
            f"{len(self.segments)} segments, {self.speech_time:.1f}s speech "
            f"of {self.duration:.1f}s, language={langs}, "
            f"speakers={len(self.speakers) or len(self.speaker_ids())}"
        )
