"""Paths, bundled tool discovery and pipeline configuration."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

# --------------------------------------------------------------------------
# Project layout (everything is self-contained / portable)
# --------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PYTHON_DIR = PROJECT_ROOT / ".python"
TOOLS_DIR = PROJECT_ROOT / ".tools"
FFMPEG_DIR = TOOLS_DIR / "ffmpeg"
FFMPEG_BIN = FFMPEG_DIR / "bin" / "ffmpeg.exe"
FFPROBE_BIN = FFMPEG_DIR / "bin" / "ffprobe.exe"

MODELS_DIR = PROJECT_ROOT / "models"
INPUT_DIR = PROJECT_ROOT / "input"
WORK_DIR = PROJECT_ROOT / "work"
OUTPUT_DIR = PROJECT_ROOT / "output"
CACHE_DIR = PROJECT_ROOT / "cache"
LOG_DIR = PROJECT_ROOT / "logs"

#: F5-TTS imports matplotlib, which aborts if it cannot write its default cache
#: directory (%LOCALAPPDATA%\matplotlib is locked down on this host).  Point it
#: somewhere writable *before* anything imports matplotlib.
_MPL_CACHE = PROJECT_ROOT / ".cache" / "matplotlib"
try:
    _MPL_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(_MPL_CACHE))
except OSError:  # pragma: no cover - non-fatal
    pass

#: HuggingFace caches must live inside the project.  Two reasons:
#:   1. the sandbox denies writes to %USERPROFILE%\.cache and to
#:      %LOCALAPPDATA%\huggingface, so the default cache paths fail;
#:   2. the Xet transfer backend (hf_xet) cannot run here at all -- it needs
#:      I/O the sandbox forbids and dies with "I/O error: Acesso negado".
#: Disabling Xet makes huggingface_hub fall back to plain HTTP downloads,
#: which work fine.
_HF_HOME = PROJECT_ROOT / ".cache" / "huggingface"
_HF_HUB_CACHE = MODELS_DIR / "hf"
_HF_XET_CACHE = PROJECT_ROOT / ".cache" / "xet"
for _p in (_HF_HOME, _HF_HUB_CACHE, _HF_XET_CACHE):
    try:
        _p.mkdir(parents=True, exist_ok=True)
    except OSError:  # pragma: no cover - non-fatal
        pass

os.environ.setdefault("HF_HOME", str(_HF_HOME))
os.environ.setdefault("HF_HUB_CACHE", str(_HF_HUB_CACHE))
os.environ.setdefault("HF_XET_CACHE", str(_HF_XET_CACHE))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))


def ensure_dirs() -> None:
    for d in (MODELS_DIR, INPUT_DIR, WORK_DIR, OUTPUT_DIR, CACHE_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def tool_env() -> Dict[str, str]:
    """Environment with the bundled ffmpeg + Python Scripts on PATH.

    Several libraries (whisper, torchaudio helpers, wav2lip) shell out to
    ffmpeg, so it must be reachable by name.
    """
    env = dict(os.environ)
    extra = [str(FFMPEG_DIR / "bin"), str(PYTHON_DIR), str(PYTHON_DIR / "Scripts")]
    env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    env["MPLCONFIGDIR"] = str(_MPL_CACHE)
    env["HF_HOME"] = str(_HF_HOME)
    env["HF_HUB_CACHE"] = str(_HF_HUB_CACHE)
    env["HF_XET_CACHE"] = str(_HF_XET_CACHE)
    env["HF_HUB_DISABLE_XET"] = "1"
    env["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
    env["XDG_CACHE_HOME"] = str(PROJECT_ROOT / ".cache")
    # Keep torch from spawning a thread per physical core on this small CPU box.
    env.setdefault("OMP_NUM_THREADS", str(max(1, (os.cpu_count() or 2))))
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    return env


# --------------------------------------------------------------------------
# Languages
# --------------------------------------------------------------------------
# Whisper language codes <-> human readable names <-> translation targets.
LANGUAGES: Dict[str, str] = {
    "af": "Afrikaans", "ar": "Arabic", "bg": "Bulgarian", "bn": "Bengali",
    "ca": "Catalan", "cs": "Czech", "da": "Danish", "de": "German",
    "el": "Greek", "en": "English", "es": "Spanish", "et": "Estonian",
    "fa": "Persian", "fi": "Finnish", "fr": "French", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "hu": "Hungarian", "id": "Indonesian",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "lt": "Lithuanian",
    "lv": "Latvian", "ms": "Malay", "nl": "Dutch", "no": "Norwegian",
    "pl": "Polish", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sk": "Slovak", "sl": "Slovenian", "sr": "Serbian", "sv": "Swedish",
    "ta": "Tamil", "th": "Thai", "tr": "Turkish", "uk": "Ukrainian",
    "ur": "Urdu", "vi": "Vietnamese", "zh": "Chinese",
}

# Languages where F5-TTS has no native grapheme coverage and therefore needs
# romanisation before synthesis.
NON_LATIN = {"zh", "ja", "ko", "ar", "he", "fa", "ur", "hi", "bn", "ta", "th", "el", "ru", "uk", "bg", "sr"}


def normalize_lang(code: str) -> str:
    """Accept 'pt', 'pt-BR', 'Portuguese', 'português' -> 'pt'."""
    if not code:
        raise ValueError("empty language code")
    c = code.strip().lower().replace("_", "-")
    if c in LANGUAGES:
        return c
    base = c.split("-")[0]
    if base in LANGUAGES:
        return base
    for k, name in LANGUAGES.items():
        if name.lower() == c:
            return k
    raise ValueError(
        f"unsupported language {code!r}. Supported: {', '.join(sorted(LANGUAGES))}"
    )


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
@dataclass
class DubladorConfig:
    """Everything the pipeline needs. Persisted next to each job."""

    input_video: str = ""
    output_video: str = ""
    target_lang: str = "pt"
    source_lang: Optional[str] = None          # None -> auto-detect

    # ---- speech to text -------------------------------------------------
    #: "auto" = let dublador.hardware pick the model for this machine.
    whisper_model: str = "auto"                # auto|tiny|base|small|medium|large-v3
    whisper_compute_type: str = "auto"         # auto|int8|int8_float32|float16|float32
    whisper_beam_size: int = 5
    vad_filter: bool = True

    # ---- translation ----------------------------------------------------
    translator: str = "google"                 # google|mymemory|libre
    translate_batch_size: int = 20

    # ---- voice analysis -------------------------------------------------
    min_speakers: int = 1
    max_speakers: int = 6
    #: Cosine-similarity cut for speaker clustering.  0 means "auto": pick the
    #: default for the active embedding backend (see
    #: ``dublador.speakers.AUTO_THRESHOLD``).  The two backends live in very
    #: different similarity scales, so a single hard-coded value is wrong for
    #: one of them.
    speaker_threshold: float = 0.0
    ref_min_seconds: float = 4.0               # minimum usable F5-TTS reference
    ref_max_seconds: float = 12.0

    # ---- text to speech -------------------------------------------------
    tts_backend: str = "f5-tts"
    f5_model: str = "F5TTS_v1_Base"
    f5_ckpt: Optional[str] = None
    f5_vocab: Optional[str] = None
    tts_speed: float = 1.0
    tts_nfe_step: int = 0                 # 0 = auto (from detected hardware)
    tts_cfg_strength: float = 2.0
    tts_target_rms: float = 0.1
    tts_cross_fade: float = 0.15
    tts_sway_coef: float = -1.0
    #: Ask F5-TTS to generate speech that already fills the original slot.
    #: This is what keeps the dub in sync with the picture without artefacts
    #: from time-stretching.
    tts_fix_duration: bool = True
    tts_remove_silence: bool = False
    tts_show_progress: bool = False
    tts_max_chars: int = 220              # split anything longer before synthesis

    # ---- timing ---------------------------------------------------------
    max_stretch: float = 1.40                  # xatempo ceiling before we re-synthesise
    min_stretch: float = 0.72
    segment_gap: float = 0.06                  # silence padding between lines

    # ---- lip sync -------------------------------------------------------
    lipsync: bool = True
    lipsync_backend: str = "wav2lip"
    lipsync_batch: int = 8
    #: Measured on this CPU-only host, Wav2Lip costs roughly one second per
    #: frame -- face detection dominates -- so a 1-minute 25 fps clip is around
    #: 25 minutes of compute.  Above this many seconds the pipeline skips lip
    #: sync automatically (and says so) unless :attr:`lipsync_force` is set.
    lipsync_max_seconds: float = 90.0
    lipsync_force: bool = False
    #: >1 shrinks frames before Wav2Lip; 2 is ~4x faster with some softness.
    lipsync_resize_factor: int = 1

    # ---- mix ------------------------------------------------------------
    #: How to treat the original soundtrack.
    #:   "none"     -> discard it, deliver a clean dub (default, safest)
    #:   "duck"     -> keep music/SFX by heavily attenuating the original
    #:                 (note: the ORIGINAL VOICES remain faintly audible,
    #:                  because we do not do source separation)
    #:   "separate" -> use Demucs to remove the original vocals, if installed
    background_mode: str = "none"
    background_gain_db: float = -11.0
    voice_gain_db: float = 1.5

    # ---- runtime --------------------------------------------------------
    device: str = "auto"                       # auto|cpu|cuda
    threads: int = 0                           # 0 -> os.cpu_count()
    seed: int = 1234
    job_name: str = "job"
    resume: bool = True

    # ---- derived --------------------------------------------------------
    extra: Dict[str, Any] = field(default_factory=dict)

    # ....................................................................
    def resolve_device(self) -> str:
        if self.device != "auto":
            return self.device
        try:
            import torch  # noqa: PLC0415

            if torch.cuda.is_available():
                return "cuda"
        except Exception:  # noqa: BLE001
            pass
        return "cpu"

    def resolve_threads(self) -> int:
        if self.threads and self.threads > 0:
            return self.threads
        return max(1, os.cpu_count() or 1)

    def job_dir(self, root: Optional[Path] = None) -> Path:
        base = Path(root) if root else WORK_DIR
        d = base / self.job_name
        d.mkdir(parents=True, exist_ok=True)
        return d

    # ....................................................................
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: Path) -> "DubladorConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def from_args(cls, **kwargs: Any) -> "DubladorConfig":
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(kwargs) - known
        if unknown:
            raise TypeError(f"unknown config keys: {sorted(unknown)}")
        return cls(**{k: v for k, v in kwargs.items() if v is not None})


def python_exe() -> str:
    """Path to the interpreter that should run pipeline sub-processes."""
    return sys.executable
