"""Wav2Lip lip-sync stage (CPU only, self-contained).

Why this module vendors the network instead of shelling out to upstream
-----------------------------------------------------------------------
The reference implementation (Rudrabha/Wav2Lip) ships ``inference.py`` which
depends on an *ancient* stack (torch 1.x, ``librosa.filters.mel`` with
positional arguments, a vendored copy of face-alignment's ``face_detection``
package).  On this machine we have Python 3.11 + torch 2.x and the PyPI
``face-detection`` project is a completely different API (hukkelas DSFD), so
running upstream's script unmodified is not viable.  Everything that matters is
therefore reproduced *in process* here:

* the Wav2Lip generator (``models/wav2lip.py`` + ``models/conv.py``),
* the mel-spectrogram front end (``audio.py`` + ``hparams.py``),
* the S3FD face detector (``face_detection/detection/sfd/*``).

The only things downloaded are the weights, into ``MODELS_DIR / "wav2lip"``.

Behaviour that differs from upstream on purpose
-----------------------------------------------
* Frames are streamed instead of being loaded into RAM (this box has ~1.3 GB
  free -- upstream would need tens of GB for a long 1080p source).  Frames are
  read twice: once for face detection, once for generation, then written
  frame-by-frame straight into an OpenCV ``VideoWriter``.
* A frame where S3FD finds no face does not abort the run: the last known box
  is reused.  Only a video with *no* detectable face at all is an error.
* Frames are written with OpenCV (DIVX/AVI, same as upstream) and the dubbed
  audio is then muxed with ffmpeg (``media.mux_video_audio``), re-encoding to
  h264/mp4 like upstream's final ``ffmpeg -q:v 1`` step.
* Face detection on very large frames runs on a downscaled copy (see
  ``_DETECT_MAX_SIDE``) because S3FD on full-resolution 1080p takes seconds per
  frame here.

Speed: this is CPU-only (Intel i3-4130, 2 cores, no CUDA) and runs *far* below
realtime -- expect roughly 1-4 generated frames per second depending on
resolution, so a 1-minute 720p clip can take several minutes.  Use
``resize_factor > 1`` to trade quality for speed, and pass a ``timeout`` if the
caller needs a hard budget.

Host constraint: this sandbox forbids subprocess stdio pipes, so every child
process goes through :func:`dublador.utils.run_command` (temp-file redirect)
and no ``subprocess`` call of our own uses ``PIPE``.
"""
from __future__ import annotations

import os
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MODELS_DIR, DubladorConfig
from .media import convert_audio, media_info, mux_video_audio
from .utils import (
    LOG,
    CommandError,
    ensure_dir,
    human_bytes,
    human_duration,
    read_json,
    write_json,
)

__all__ = [
    "lipsync_available",
    "ensure_models",
    "lipsync_video",
    "lipsync_fallback",
    "WAV2LIP_GAN_URLS",
    "S3FD_URLS",
    "lipsync_model_dir",
]

# --------------------------------------------------------------------------
# Model weights
# --------------------------------------------------------------------------
WAV2LIP_SUBDIR = "wav2lip"
WAV2LIP_GAN_NAME = "wav2lip_gan.pth"
WAV2LIP_NAME = "wav2lip.pth"
S3FD_NAME = "s3fd.pth"

#: ``wav2lip_gan`` (~436 MB) is the higher quality generator.
WAV2LIP_GAN_URLS: Tuple[str, ...] = (
    "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip_gan.pth",
    "https://huggingface.co/Non-playing-Character/Wave2lip/resolve/main/wav2lip_gan.pth",
    "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth",
)

#: Lighter, slightly softer alternative (same architecture).
WAV2LIP_URLS: Tuple[str, ...] = (
    "https://huggingface.co/camenduru/Wav2Lip/resolve/main/checkpoints/wav2lip.pth",
)

#: S3FD face detector used by Wav2Lip (the official face-alignment mirror first).
S3FD_URLS: Tuple[str, ...] = (
    "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
    "https://huggingface.co/spaces/zmbfeng/text_to_speech_sync_video/resolve/main/Wav2Lip/face_detection/detection/sfd/s3fd.pth",
)

#: Below these sizes a download is considered truncated / an error page.
_MIN_BYTES: Dict[str, int] = {
    WAV2LIP_GAN_NAME: 300 * 1024 * 1024,
    WAV2LIP_NAME: 300 * 1024 * 1024,
    S3FD_NAME: 60 * 1024 * 1024,
}

_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) dublador-bootstrap/1.0"

# --------------------------------------------------------------------------
# Audio front end hyper-parameters (upstream hparams.py, inference subset)
# --------------------------------------------------------------------------
_NUM_MELS = 80
_N_FFT = 800
_HOP_SIZE = 200
_WIN_SIZE = 800
_SAMPLE_RATE = 16000
_FMIN = 55.0
_FMAX = 7600.0
_PREEMPHASIS = 0.97
_REF_LEVEL_DB = 20.0
_MIN_LEVEL_DB = -100.0
_MAX_ABS_VALUE = 4.0

_IMG_SIZE = 96
_MEL_STEP_SIZE = 16
_SMOOTH_WINDOW = 5

#: S3FD runs on the *full* resolution frame, which costs seconds per frame on
#: this CPU box at 1080p+.  Frames whose long side exceeds this are only
#: downscaled *for the detector* and the boxes are scaled back, which keeps the
#: crop the generator sees identical.  Set to 0 to disable and be byte-exact
#: with upstream.
_DETECT_MAX_SIDE = 960
_detect_notice_logged = False


# --------------------------------------------------------------------------
# Availability / weights
# --------------------------------------------------------------------------
def lipsync_available() -> bool:
    """True when the packages Wav2Lip inference needs are importable.

    The weights are *not* checked here -- :func:`ensure_models` downloads them.
    """
    try:
        import cv2  # noqa: F401,PLC0415
        import librosa  # noqa: F401,PLC0415
        import torch  # noqa: F401,PLC0415
    except Exception as e:  # noqa: BLE001
        LOG.debug("lipsync unavailable: %s", e)
        return False
    return True


def lipsync_model_dir(cfg: Optional[DubladorConfig] = None) -> Path:
    """Directory holding the Wav2Lip weights (``models/wav2lip``)."""
    override = None
    if cfg is not None:
        override = (cfg.extra or {}).get("lipsync_model_dir")
    return Path(override) if override else (MODELS_DIR / WAV2LIP_SUBDIR)


def _looks_complete(path: Path, min_bytes: int) -> bool:
    try:
        return path.is_file() and path.stat().st_size >= min_bytes
    except OSError:
        return False


def _http_download(url: str, dest: Path, *, timeout: float = 120.0) -> int:
    """Stream ``url`` into ``dest`` (+ ``.part`` then atomic rename).

    Pure Python / in-process, so nothing is piped through a shell.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    got = 0
    total = 0
    last_pct = -1
    LOG.info("  downloading %s", url)
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
        total = int(resp.headers.get("Content-Length") or 0)
        while True:
            chunk = resp.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            got += len(chunk)
            if total:
                pct = int(got * 100 / total)
                if pct >= last_pct + 10:
                    last_pct = pct
                    LOG.info("    ...%d%% (%s / %s)", pct, human_bytes(got), human_bytes(total))
    if got == 0:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"empty response for {url}")
    if total and got < total:
        part.unlink(missing_ok=True)
        raise RuntimeError(f"truncated download from {url}: {got} of {total} bytes")
    part.replace(dest)
    return got


def _torch_load(path: Path) -> Any:
    """``torch.load`` with a fallback for checkpoints torch refuses by default."""
    import torch  # noqa: PLC0415

    try:
        return torch.load(str(path), map_location="cpu")
    except Exception:  # noqa: BLE001
        return torch.load(str(path), map_location="cpu", weights_only=False)


def _verify_state_dict(path: Path) -> str:
    """Open ``path`` with torch and describe the state dict inside it."""
    try:
        import torch  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        return "size only (torch not importable)"
    obj = _torch_load(path)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if isinstance(obj, dict) and len(obj) > 0:
        return f"{len(obj)} tensors"
    raise RuntimeError(f"{path.name} does not contain a usable state dict")


def _ensure_weight(
    dest: Path,
    urls: Sequence[str],
    *,
    what: str,
    deep: bool = True,
) -> Path:
    """Make sure ``dest`` holds a complete, loadable checkpoint."""
    min_bytes = _MIN_BYTES.get(dest.name, 1024)
    marker = dest.with_name(dest.name + ".ok")
    if _looks_complete(dest, min_bytes):
        rec = read_json(marker, default=None)
        size = dest.stat().st_size
        if isinstance(rec, dict) and rec.get("size") == size:
            return dest
        if not deep:
            return dest
        LOG.info("%s: verifying existing %s", what, dest.name)
        detail = _verify_state_dict(dest)
        write_json(marker, {"size": size, "detail": detail, "at": time.strftime("%Y-%m-%d %H:%M:%S")})
        LOG.info("%s: %s OK (%s, %s)", what, dest.name, human_bytes(size), detail)
        return dest

    errors: List[str] = []
    for url in urls:
        try:
            got = _http_download(url, dest)
        except Exception as e:  # noqa: BLE001
            LOG.warning("%s: download failed from %s (%s)", what, url, e)
            errors.append(f"{url}: {e}")
            continue
        if not _looks_complete(dest, min_bytes):
            LOG.warning("%s: %s looks truncated (%s)", what, dest.name, human_bytes(got))
            dest.unlink(missing_ok=True)
            errors.append(f"{url}: only {got} bytes")
            continue
        try:
            detail = _verify_state_dict(dest) if deep else "size only"
        except Exception as e:  # noqa: BLE001
            LOG.warning("%s: %s failed torch.load (%s)", what, dest.name, e)
            dest.unlink(missing_ok=True)
            errors.append(f"{url}: torch.load failed: {e}")
            continue
        size = dest.stat().st_size
        write_json(marker, {"size": size, "detail": detail, "url": url})
        LOG.info("%s: %s ready (%s, %s)", what, dest.name, human_bytes(size), detail)
        return dest

    raise RuntimeError(
        f"could not download {dest.name} ({what}); tried {len(urls)} mirror(s):\n  "
        + "\n  ".join(errors)
    )


def ensure_models(cfg: Optional[DubladorConfig] = None) -> Path:
    """Download the Wav2Lip weights if needed and return the model directory."""
    models = ensure_dir(lipsync_model_dir(cfg))
    LOG.info("lipsync models: %s", models)

    checkpoint = models / WAV2LIP_GAN_NAME
    try:
        _ensure_weight(checkpoint, WAV2LIP_GAN_URLS, what="wav2lip-gan")
    except RuntimeError:
        # Fall back to the non-GAN generator if every GAN mirror is dead.
        checkpoint = models / WAV2LIP_NAME
        _ensure_weight(checkpoint, WAV2LIP_URLS, what="wav2lip")

    _ensure_weight(models / S3FD_NAME, S3FD_URLS, what="s3fd")
    return models


def _pick_checkpoint(models: Path) -> Path:
    for name in (WAV2LIP_GAN_NAME, WAV2LIP_NAME):
        p = models / name
        if _looks_complete(p, _MIN_BYTES.get(name, 1024)):
            return p
    raise RuntimeError(
        f"no Wav2Lip checkpoint in {models} (expected {WAV2LIP_GAN_NAME} or {WAV2LIP_NAME})"
    )


# --------------------------------------------------------------------------
# Mel front end (vendored from Wav2Lip audio.py + hparams.py)
# --------------------------------------------------------------------------
def _melspectrogram(wav: np.ndarray) -> np.ndarray:
    """80-bin, 16 kHz log-mel, normalised to [-4, 4] exactly like upstream."""
    import librosa  # noqa: PLC0415
    from scipy import signal  # noqa: PLC0415

    wav = signal.lfilter([1, -_PREEMPHASIS], [1], wav)
    # NOTE: modern librosa made sr/n_fft keyword-only, hence the keywords.
    d = librosa.stft(y=wav, n_fft=_N_FFT, hop_length=_HOP_SIZE, win_length=_WIN_SIZE)
    mel_basis = librosa.filters.mel(
        sr=_SAMPLE_RATE, n_fft=_N_FFT, n_mels=_NUM_MELS, fmin=_FMIN, fmax=_FMAX
    )
    s = np.dot(mel_basis, np.abs(d))

    min_level = np.exp(_MIN_LEVEL_DB / 20 * np.log(10))
    s = 20.0 * np.log10(np.maximum(min_level, s)) - _REF_LEVEL_DB

    # allow_clipping_in_normalization=True + symmetric_mels=True
    return np.clip(
        (2 * _MAX_ABS_VALUE) * ((s - _MIN_LEVEL_DB) / (-_MIN_LEVEL_DB)) - _MAX_ABS_VALUE,
        -_MAX_ABS_VALUE,
        _MAX_ABS_VALUE,
    )


def _load_wav_16k(audio: Path, tmp_wav: Path) -> np.ndarray:
    """Load any audio file as 16 kHz mono float32 (via ffmpeg, then librosa)."""
    import librosa  # noqa: PLC0415

    convert_audio(audio, tmp_wav, sample_rate=_SAMPLE_RATE, channels=1)
    wav, _sr = librosa.load(str(tmp_wav), sr=_SAMPLE_RATE)
    return np.asarray(wav, dtype=np.float32)


def _mel_chunk_count(mel_cols: int, fps: float) -> int:
    """How many 16-column mel windows upstream would produce for this audio."""
    if not fps or fps != fps or fps <= 0:
        raise RuntimeError(f"lipsync: unusable video fps ({fps!r})")
    multiplier = 80.0 / fps
    i = 0
    while True:
        start = int(i * multiplier)
        if start + _MEL_STEP_SIZE > mel_cols:
            return i + 1
        i += 1


def _mel_chunk(mel: np.ndarray, index: int, fps: float) -> np.ndarray:
    start = int(index * (80.0 / fps))
    if start + _MEL_STEP_SIZE > mel.shape[1]:
        return mel[:, mel.shape[1] - _MEL_STEP_SIZE:]
    return mel[:, start:start + _MEL_STEP_SIZE]


# --------------------------------------------------------------------------
# Wav2Lip generator (vendored from models/wav2lip.py + models/conv.py)
# --------------------------------------------------------------------------
def _configure_torch_threads(torch) -> None:
    """Give torch every logical core; the default is physical cores only.

    This is a 2-core i3, so the extra hyper-threads are worth ~20-30% on the
    conv-heavy detector and generator.
    """
    want = max(1, min(8, os.cpu_count() or 2))
    if torch.get_num_threads() != want:
        try:
            torch.set_num_threads(want)
        except Exception:  # noqa: BLE001
            LOG.debug("could not raise torch threads", exc_info=True)


def _build_wav2lip():
    """Instantiate the generator.  Defined here so ``import torch`` stays lazy."""
    import torch  # noqa: PLC0415
    from torch import nn  # noqa: PLC0415

    _configure_torch_threads(torch)

    class _Conv2d(nn.Module):
        def __init__(self, cin, cout, kernel_size, stride, padding, residual=False):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size, stride, padding),
                nn.BatchNorm2d(cout),
            )
            self.act = nn.ReLU()
            self.residual = residual

        def forward(self, x):
            out = self.conv_block(x)
            if self.residual:
                out = out + x
            return self.act(out)

    class _Conv2dTranspose(nn.Module):
        def __init__(self, cin, cout, kernel_size, stride, padding, output_padding=0):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.ConvTranspose2d(cin, cout, kernel_size, stride, padding, output_padding),
                nn.BatchNorm2d(cout),
            )
            self.act = nn.ReLU()

        def forward(self, x):
            return self.act(self.conv_block(x))

    class _Wav2Lip(nn.Module):
        def __init__(self):
            super().__init__()
            self.face_encoder_blocks = nn.ModuleList([
                nn.Sequential(_Conv2d(6, 16, kernel_size=7, stride=1, padding=3)),
                nn.Sequential(
                    _Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                    _Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                    _Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    _Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    _Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    _Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    _Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    _Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    _Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                    _Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    _Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
                    _Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
                ),
            ])
            self.audio_encoder = nn.Sequential(
                _Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
                _Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                _Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
                _Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
                _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                _Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
                _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                _Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
                _Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                _Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
                _Conv2d(512, 512, kernel_size=1, stride=1, padding=0),
            )
            self.face_decoder_blocks = nn.ModuleList([
                nn.Sequential(_Conv2d(512, 512, kernel_size=1, stride=1, padding=0)),
                nn.Sequential(
                    _Conv2dTranspose(1024, 512, kernel_size=3, stride=1, padding=0),
                    _Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    _Conv2dTranspose(1024, 512, kernel_size=3, stride=2, padding=1, output_padding=1),
                    _Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                    _Conv2d(512, 512, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    _Conv2dTranspose(768, 384, kernel_size=3, stride=2, padding=1, output_padding=1),
                    _Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                    _Conv2d(384, 384, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    _Conv2dTranspose(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1),
                    _Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                    _Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    _Conv2dTranspose(320, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
                    _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                    _Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
                ),
                nn.Sequential(
                    _Conv2dTranspose(160, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                    _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                    _Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
                ),
            ])
            self.output_block = nn.Sequential(
                _Conv2d(80, 32, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(32, 3, kernel_size=1, stride=1, padding=0),
                nn.Sigmoid(),
            )

        def forward(self, audio_sequences, face_sequences):
            b = audio_sequences.size(0)
            dim = len(face_sequences.size())
            if dim > 4:
                audio_sequences = torch.cat(
                    [audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0
                )
                face_sequences = torch.cat(
                    [face_sequences[:, :, i] for i in range(face_sequences.size(2))], dim=0
                )

            audio_embedding = self.audio_encoder(audio_sequences)

            feats = []
            x = face_sequences
            for f in self.face_encoder_blocks:
                x = f(x)
                feats.append(x)

            x = audio_embedding
            for f in self.face_decoder_blocks:
                x = f(x)
                x = torch.cat((x, feats[-1]), dim=1)
                feats.pop()

            x = self.output_block(x)

            if dim > 4:
                x = torch.split(x, b, dim=0)
                return torch.stack(x, dim=2)
            return x

    return _Wav2Lip()


def _load_generator(checkpoint: Path, device: str):
    """Load the Wav2Lip generator, stripping any ``module.`` DataParallel prefix."""
    import torch  # noqa: PLC0415

    model = _build_wav2lip()
    state = _torch_load(checkpoint)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {k.replace("module.", ""): v for k, v in state.items()}
    try:
        model.load_state_dict(cleaned)
    except RuntimeError as e:
        raise RuntimeError(
            f"Wav2Lip checkpoint {checkpoint.name} does not match the network "
            f"({e}); delete it and re-run ensure_models()"
        ) from e
    return model.to(device).eval()


# --------------------------------------------------------------------------
# S3FD face detector (vendored from face_detection/detection/sfd/*)
# --------------------------------------------------------------------------
def _build_s3fd():
    """The s3fd (SFD) detector network."""
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    import torch.nn.functional as f  # noqa: PLC0415

    class _L2Norm(nn.Module):
        def __init__(self, n_channels, scale=1.0):
            super().__init__()
            self.n_channels = n_channels
            self.scale = scale
            self.eps = 1e-10
            self.weight = nn.Parameter(torch.Tensor(self.n_channels))
            self.weight.data *= 0.0
            self.weight.data += self.scale

        def forward(self, x):
            norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
            return x / norm * self.weight.view(1, -1, 1, 1)

    class _S3FD(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
            self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

            self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
            self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

            self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
            self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

            self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
            self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

            self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
            self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

            self.fc6 = nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=3)
            self.fc7 = nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0)

            self.conv6_1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
            self.conv6_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)

            self.conv7_1 = nn.Conv2d(512, 128, kernel_size=1, stride=1, padding=0)
            self.conv7_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)

            self.conv3_3_norm = _L2Norm(256, scale=10)
            self.conv4_3_norm = _L2Norm(512, scale=8)
            self.conv5_3_norm = _L2Norm(512, scale=5)

            self.conv3_3_norm_mbox_conf = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
            self.conv3_3_norm_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)
            self.conv4_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
            self.conv4_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
            self.conv5_3_norm_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
            self.conv5_3_norm_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)

            self.fc7_mbox_conf = nn.Conv2d(1024, 2, kernel_size=3, stride=1, padding=1)
            self.fc7_mbox_loc = nn.Conv2d(1024, 4, kernel_size=3, stride=1, padding=1)
            self.conv6_2_mbox_conf = nn.Conv2d(512, 2, kernel_size=3, stride=1, padding=1)
            self.conv6_2_mbox_loc = nn.Conv2d(512, 4, kernel_size=3, stride=1, padding=1)
            self.conv7_2_mbox_conf = nn.Conv2d(256, 2, kernel_size=3, stride=1, padding=1)
            self.conv7_2_mbox_loc = nn.Conv2d(256, 4, kernel_size=3, stride=1, padding=1)

        def forward(self, x):
            h = f.relu(self.conv1_1(x))
            h = f.relu(self.conv1_2(h))
            h = f.max_pool2d(h, 2, 2)

            h = f.relu(self.conv2_1(h))
            h = f.relu(self.conv2_2(h))
            h = f.max_pool2d(h, 2, 2)

            h = f.relu(self.conv3_1(h))
            h = f.relu(self.conv3_2(h))
            h = f.relu(self.conv3_3(h))
            f3_3 = h
            h = f.max_pool2d(h, 2, 2)

            h = f.relu(self.conv4_1(h))
            h = f.relu(self.conv4_2(h))
            h = f.relu(self.conv4_3(h))
            f4_3 = h
            h = f.max_pool2d(h, 2, 2)

            h = f.relu(self.conv5_1(h))
            h = f.relu(self.conv5_2(h))
            h = f.relu(self.conv5_3(h))
            f5_3 = h
            h = f.max_pool2d(h, 2, 2)

            h = f.relu(self.fc6(h))
            h = f.relu(self.fc7(h))
            ffc7 = h
            h = f.relu(self.conv6_1(h))
            h = f.relu(self.conv6_2(h))
            f6_2 = h
            h = f.relu(self.conv7_1(h))
            h = f.relu(self.conv7_2(h))
            f7_2 = h

            f3_3 = self.conv3_3_norm(f3_3)
            f4_3 = self.conv4_3_norm(f4_3)
            f5_3 = self.conv5_3_norm(f5_3)

            cls1 = self.conv3_3_norm_mbox_conf(f3_3)
            reg1 = self.conv3_3_norm_mbox_loc(f3_3)
            cls2 = self.conv4_3_norm_mbox_conf(f4_3)
            reg2 = self.conv4_3_norm_mbox_loc(f4_3)
            cls3 = self.conv5_3_norm_mbox_conf(f5_3)
            reg3 = self.conv5_3_norm_mbox_loc(f5_3)
            cls4 = self.fc7_mbox_conf(ffc7)
            reg4 = self.fc7_mbox_loc(ffc7)
            cls5 = self.conv6_2_mbox_conf(f6_2)
            reg5 = self.conv6_2_mbox_loc(f6_2)
            cls6 = self.conv7_2_mbox_conf(f7_2)
            reg6 = self.conv7_2_mbox_loc(f7_2)

            chunk = torch.chunk(cls1, 4, 1)
            bmax = torch.max(torch.max(chunk[0], chunk[1]), chunk[2])
            cls1 = torch.cat([bmax, chunk[3]], dim=1)

            return [cls1, reg1, cls2, reg2, cls3, reg3, cls4, reg4, cls5, reg5, cls6, reg6]

    return _S3FD()


def _nms(dets: np.ndarray, thresh: float) -> List[int]:
    if dets is None or len(dets) == 0:
        return []
    x1, y1, x2, y2, scores = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: List[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        denom = areas[i] + areas[order[1:]] - w * h
        with np.errstate(divide="ignore", invalid="ignore"):
            ovr = np.where(denom > 0, w * h / denom, 0.0)
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def _batch_decode(loc, priors, variances):
    import torch  # noqa: PLC0415

    boxes = torch.cat((
        priors[:, :, :2] + loc[:, :, :2] * variances[0] * priors[:, :, 2:],
        priors[:, :, 2:] * torch.exp(loc[:, :, 2:] * variances[1]),
    ), 2)
    boxes[:, :, :2] -= boxes[:, :, 2:] / 2
    boxes[:, :, 2:] += boxes[:, :, :2]
    return boxes


def _batch_detect(net, images: np.ndarray, device: str) -> Tuple[np.ndarray, int]:
    """S3FD forward + anchor decoding for a batch of RGB uint8 images.

    Returns ``(bboxlist, batch)`` with ``bboxlist`` shaped ``(anchors, batch, 5)``.
    """
    import torch  # noqa: PLC0415
    import torch.nn.functional as f  # noqa: PLC0415

    n = images.shape[0]
    x = images.astype(np.float32) - np.array([104, 117, 123], dtype=np.float32)
    x = np.ascontiguousarray(x.transpose(0, 3, 1, 2))

    tensor = torch.from_numpy(x).to(device)
    with torch.no_grad():
        olist = [o.detach().cpu() for o in net(tensor)]
    del tensor

    for i in range(len(olist) // 2):
        olist[i * 2] = f.softmax(olist[i * 2], dim=1)

    bboxlist = []
    for i in range(len(olist) // 2):
        ocls, oreg = olist[i * 2], olist[i * 2 + 1]
        stride = 2 ** (i + 2)  # 4, 8, 16, 32, 64, 128
        poss = zip(*np.where(ocls[:, 1, :, :] > 0.05))
        for _i, hindex, windex in poss:
            axc = stride / 2 + windex * stride
            ayc = stride / 2 + hindex * stride
            score = ocls[:, 1, hindex, windex]
            loc = oreg[:, :, hindex, windex].contiguous().view(n, 1, 4)
            priors = torch.Tensor([[axc, ayc, stride * 4, stride * 4]]).view(1, 1, 4)
            box = _batch_decode(loc, priors, [0.1, 0.2])[:, 0] * 1.0
            bboxlist.append(torch.cat([box, score.unsqueeze(1)], 1).cpu().numpy())

    if not bboxlist:
        return np.zeros((1, n, 5), dtype=np.float32), n
    return np.array(bboxlist), n


class _FaceDetector:
    """Thin S3FD wrapper returning one face rect per frame.

    Mirrors ``face_detection.api.FaceAlignment.get_detections_for_batch``:
    BGR frames in, ``(x1, y1, x2, y2)`` (or ``None``) out.
    """

    def __init__(self, weights: Path, device: str = "cpu") -> None:
        import torch  # noqa: PLC0415

        _configure_torch_threads(torch)
        self.device = device
        self._torch = torch
        self.net = _build_s3fd()
        state = _torch_load(weights)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        self.net.load_state_dict({k.replace("module.", ""): v for k, v in state.items()})
        self.net.to(device).eval()

    def detect_batch(self, images: np.ndarray) -> List[Optional[Tuple[int, int, int, int]]]:
        """``images`` is (B, H, W, 3) uint8 BGR (as read by OpenCV)."""
        rgb = np.ascontiguousarray(images[..., ::-1])
        bboxlists, n = _batch_detect(self.net, rgb, self.device)

        results: List[Optional[Tuple[int, int, int, int]]] = []
        for i in range(n):
            keep = _nms(bboxlists[:, i, :], 0.3)
            if keep:
                dets = bboxlists[keep, i, :]
                dets = [d for d in dets if d[-1] > 0.5]
            else:
                dets = []
            if not dets:
                results.append(None)
                continue
            d = np.clip(dets[0], 0, None)
            x1, y1, x2, y2 = map(int, d[:-1])
            results.append((x1, y1, x2, y2))
        return results

    def __del__(self) -> None:  # keep peak RSS down between runs
        self.net = None


# --------------------------------------------------------------------------
# Video helpers
# --------------------------------------------------------------------------
def _open_capture(path: Path, cv2) -> Any:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"lipsync: OpenCV cannot open video {path}")
    return cap


def _read_frame(cap, cv2, resize_factor: int) -> Optional[np.ndarray]:
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if resize_factor and resize_factor > 1:
        frame = cv2.resize(
            frame, (frame.shape[1] // resize_factor, frame.shape[0] // resize_factor)
        )
    return frame


def _smooth_boxes(boxes: np.ndarray, window: int = _SMOOTH_WINDOW) -> np.ndarray:
    """Temporal smoothing, byte-for-byte the upstream ``get_smoothened_boxes``."""
    boxes = np.asarray(boxes, dtype=np.float32)
    n = len(boxes)
    if n == 0:
        return boxes
    for i in range(n):
        if i + window > n:
            chunk = boxes[n - window:]
        else:
            chunk = boxes[i:i + window]
        boxes[i] = np.mean(chunk, axis=0)
    return boxes


def _fill_missing_boxes(
    raw: List[Optional[Tuple[int, int, int, int]]],
    shapes: List[Tuple[int, int]],
) -> np.ndarray:
    """Replace undetected frames with the nearest known box.

    Upstream aborts the whole run; on real footage a handful of frames without a
    face is normal, so we carry the previous box forward (and back-fill the
    leading misses from the first hit).
    """
    hits = [i for i, b in enumerate(raw) if b is not None]
    if not hits:
        raise RuntimeError(
            f"no face detected in any of the {len(raw)} sampled frame(s); use "
            "footage where the face is clearly visible (or lower resize_factor), "
            "otherwise the caller should fall back to lipsync_fallback()"
        )
    missed = len(raw) - len(hits)
    if missed:
        LOG.warning(
            "lipsync: no face found in %d/%d frame(s); reusing the nearest detection",
            missed, len(raw),
        )

    first = hits[0]
    boxes: List[Tuple[float, float, float, float]] = []
    last = raw[first]
    assert last is not None
    for i in range(len(raw)):
        cur = raw[i]
        if cur is not None:
            last = cur
        h, w = shapes[i]
        x1, y1, x2, y2 = (int(v) for v in last)
        x1 = max(0, min(x1, w - 2))
        y1 = max(0, min(y1, h - 2))
        x2 = max(x1 + 1, min(x2, w))
        y2 = max(y1 + 1, min(y2, h))
        boxes.append((x1, y1, x2, y2))
    return np.asarray(boxes, dtype=np.float32)


def _emit_progress(cb: Optional[Callable[..., Any]], done: int, total: int) -> None:
    if cb is None:
        return
    try:
        cb(done, total)
    except TypeError:
        try:
            cb(f"lipsync {done}/{total}")
        except Exception:  # noqa: BLE001
            LOG.debug("lipsync progress callback raised", exc_info=True)
    except Exception:  # noqa: BLE001
        LOG.debug("lipsync progress callback raised", exc_info=True)


def _check_timeout(started: float, timeout: Optional[float], what: str) -> None:
    if timeout is None:
        return
    elapsed = time.time() - started
    if elapsed > timeout:
        raise RuntimeError(
            f"lipsync: timed out after {human_duration(elapsed)} during {what} "
            f"(limit {human_duration(timeout)})"
        )


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------
def lipsync_video(
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    *,
    device: str = "cpu",
    batch_size: int = 8,
    static: bool = False,
    pads: Tuple[int, int, int, int] = (0, 10, 0, 0),
    resize_factor: int = 1,
    nosmooth: bool = False,
    timeout: Optional[float] = None,
    progress_cb: Optional[Callable[..., Any]] = None,
) -> Path:
    """Re-animate ``video_path``'s mouth to match ``audio_path`` with Wav2Lip.

    Raises ``RuntimeError`` (with a human readable reason) on any failure; the
    pipeline decides whether to degrade to :func:`lipsync_fallback`.
    """
    started = time.time()
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    out_path = Path(out_path)

    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"lipsync: video not found or empty: {video_path}")
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"lipsync: audio not found or empty: {audio_path}")
    if not lipsync_available():
        raise RuntimeError(
            "lipsync: Wav2Lip dependencies missing (need torch, opencv-python-headless "
            "and librosa in the portable interpreter)"
        )

    LOG.info("lipsync: Wav2Lip (device=%s, batch=%d, resize_factor=%d, static=%s)",
             device, batch_size, resize_factor, static)

    models = ensure_models()
    checkpoint = _pick_checkpoint(models)
    s3fd_weights = models / S3FD_NAME
    if not _looks_complete(s3fd_weights, _MIN_BYTES[S3FD_NAME]):
        raise RuntimeError(f"lipsync: S3FD weights missing at {s3fd_weights}")

    import cv2  # noqa: PLC0415
    import torch  # noqa: PLC0415

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_wav = out_path.with_name(out_path.stem + ".lipsync.16k.wav")
    raw_video = out_path.with_name(out_path.stem + ".lipsync.raw.avi")
    batch_size = max(1, int(batch_size))
    pad_top, pad_bottom, pad_left, pad_right = (int(v) for v in pads)

    try:
        # ---- audio -> mel -------------------------------------------------
        wav = _load_wav_16k(audio_path, tmp_wav)
        if wav.size < _SAMPLE_RATE // 5:
            raise RuntimeError("lipsync: audio shorter than 0.2 s, nothing to sync")
        mel = _melspectrogram(wav)
        if mel.shape[1] < _MEL_STEP_SIZE:
            raise RuntimeError("lipsync: audio too short to build a mel window")
        if not np.isfinite(mel).all():
            raise RuntimeError(
                "lipsync: mel spectrogram contains NaN/Inf; the dubbed track is "
                "probably silent or corrupt"
            )
        mel = np.ascontiguousarray(mel, dtype=np.float32)

        # ---- video metadata ----------------------------------------------
        probe = media_info(video_path)
        cap = _open_capture(video_path, cv2)
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        finally:
            cap.release()
        if not fps or fps != fps or fps <= 0:
            fps = probe.fps or 25.0
            LOG.warning("lipsync: OpenCV reported no fps, using %.3f", fps)

        n_chunks = _mel_chunk_count(mel.shape[1], fps)
        LOG.info(
            "lipsync: %.2f s of speech -> %d frame(s) at %.3f fps (source %s)",
            wav.size / _SAMPLE_RATE, n_chunks, fps, probe.resolution or "?",
        )

        # ---- pass 1: face detection --------------------------------------
        detector = _FaceDetector(s3fd_weights, device=device)
        raw_boxes: List[Optional[Tuple[int, int, int, int]]] = []
        shapes: List[Tuple[int, int]] = []
        cap = _open_capture(video_path, cv2)
        try:
            if static:
                frame = _read_frame(cap, cv2, resize_factor)
                if frame is None:
                    raise RuntimeError("lipsync: could not read the first video frame")
                shapes.append(frame.shape[:2])
                raw_boxes.extend(_detect_chunk(detector, [frame], pads))
                LOG.info("lipsync: static mode, using the first frame only")
            else:
                pending: List[np.ndarray] = []
                while len(raw_boxes) < n_chunks:
                    _check_timeout(started, timeout, "face detection")
                    frame = _read_frame(cap, cv2, resize_factor)
                    if frame is None:
                        break
                    pending.append(frame)
                    if len(pending) >= batch_size:
                        raw_boxes.extend(_detect_chunk(detector, pending, pads))
                        shapes.extend(f.shape[:2] for f in pending)
                        pending = []
                        LOG.info("  detect %d/%d", len(raw_boxes), n_chunks)
                if pending:
                    raw_boxes.extend(_detect_chunk(detector, pending, pads))
                    shapes.extend(f.shape[:2] for f in pending)
        finally:
            cap.release()

        n_available = len(raw_boxes)
        if n_available == 0:
            raise RuntimeError("lipsync: the video contains no decodable frames")
        wrap = n_available < n_chunks
        if wrap:
            LOG.warning(
                "lipsync: video is shorter than the dubbed audio (%d < %d frames); "
                "frames will repeat", n_available, n_chunks,
            )
        boxes = _fill_missing_boxes(raw_boxes, shapes)
        if not nosmooth and len(boxes) > 1:
            boxes = _smooth_boxes(boxes)
        LOG.info("lipsync: faces located in %d frame(s)", len(boxes))

        # ---- pass 2: generate --------------------------------------------
        model = _load_generator(checkpoint, device)
        _check_timeout(started, timeout, "model load")

        writer = None
        cap = _open_capture(video_path, cv2)
        pos = 0
        try:
            done = 0
            while done < n_chunks:
                _check_timeout(started, timeout, "inference")
                take = min(batch_size, n_chunks - done)
                frames: List[np.ndarray] = []
                coords: List[Tuple[int, int, int, int]] = []
                for k in range(take):
                    src = done + k
                    if static or wrap:
                        src = 0 if static else src % n_available
                    frame = _read_sequential(cap, cv2, resize_factor, src, pos)
                    if frame is None:
                        break
                    pos = src + 1
                    frames.append(frame)
                    coords.append(tuple(int(v) for v in boxes[0 if static else (src % n_available)]))
                if not frames:
                    break

                img_batch = np.asarray(
                    [cv2.resize(f[y1:y2, x1:x2], (_IMG_SIZE, _IMG_SIZE)) for f, (x1, y1, x2, y2)
                     in zip(frames, coords)]
                )
                mel_batch = np.asarray(
                    [_mel_chunk(mel, done + k, fps) for k in range(len(frames))]
                ).reshape(len(frames), _NUM_MELS, _MEL_STEP_SIZE, 1)

                masked = img_batch.copy()
                masked[:, _IMG_SIZE // 2:] = 0
                img_in = np.concatenate((masked, img_batch), axis=3) / 255.0

                with torch.no_grad():
                    pred = model(
                        torch.from_numpy(np.ascontiguousarray(
                            mel_batch.transpose(0, 3, 1, 2), dtype=np.float32)).to(device),
                        torch.from_numpy(np.ascontiguousarray(
                            img_in.transpose(0, 3, 1, 2), dtype=np.float32)).to(device),
                    )
                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

                for p, frame, (x1, y1, x2, y2) in zip(pred, frames, coords):
                    if x2 <= x1 or y2 <= y1:
                        continue
                    patch = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    frame[y1:y2, x1:x2] = patch
                    if writer is None:
                        h, w = frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(raw_video), cv2.VideoWriter_fourcc(*"DIVX"), fps, (w, h)
                        )
                        if not writer.isOpened():
                            raise RuntimeError(
                                f"lipsync: OpenCV cannot write {raw_video} (DIVX/AVI)"
                            )
                    writer.write(frame)

                done += len(frames)
                _emit_progress(progress_cb, done, n_chunks)
                if done % max(batch_size, 1) == 0 or done >= n_chunks:
                    LOG.info(
                        "  wav2lip %d/%d frames (%.0f%%, %s elapsed)",
                        done, n_chunks, 100.0 * done / n_chunks,
                        human_duration(time.time() - started),
                    )
        finally:
            cap.release()
            if writer is not None:
                writer.release()

        if writer is None or not raw_video.is_file() or raw_video.stat().st_size == 0:
            raise RuntimeError("lipsync: Wav2Lip produced no output frames")

        # ---- mux the dubbed audio back on --------------------------------
        # Upstream re-encodes at this point (``ffmpeg -q:v 1``); doing the same
        # turns OpenCV's mpeg4/AVI into a universally playable h264/mp4.  On a
        # slow CPU the fallback to a straight stream copy keeps it cheap.
        LOG.info("lipsync: muxing dubbed audio onto %d generated frames", done)
        try:
            mux_video_audio(raw_video, audio_path, out_path, copy_video=False,
                            preset="veryfast", crf=20, shortest=False)
        except CommandError as e:
            LOG.warning("lipsync: h264 encode failed (%s); falling back to stream copy",
                        str(e).splitlines()[0])
            mux_video_audio(raw_video, audio_path, out_path, copy_video=True,
                            shortest=False)

        if not out_path.is_file() or out_path.stat().st_size == 0:
            raise RuntimeError(f"lipsync: muxing produced no output at {out_path}")

        LOG.info("lipsync: done in %s -> %s", human_duration(time.time() - started), out_path)
        return out_path

    except (CommandError, RuntimeError) as e:
        LOG.error("lipsync failed: %s", e)
        raise RuntimeError(f"Wav2Lip lip sync failed: {e}") from e
    except Exception as e:  # noqa: BLE001
        LOG.error("lipsync failed: %s: %s", type(e).__name__, e)
        raise RuntimeError(f"Wav2Lip lip sync failed: {type(e).__name__}: {e}") from e
    finally:
        for tmp in (tmp_wav, raw_video):
            try:
                tmp.unlink()
            except OSError:
                pass


def _apply_pads(
    rect: Optional[Tuple[int, int, int, int]],
    height: int,
    width: int,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
) -> Optional[Tuple[int, int, int, int]]:
    """Upstream ``face_detect`` padding + clamping for one frame."""
    if rect is None:
        return None
    x1, y1, x2, y2 = rect
    y1 = max(0, y1 - pad_top)
    y2 = min(height, y2 + pad_bottom)
    x1 = max(0, x1 - pad_left)
    x2 = min(width, x2 + pad_right)
    if x2 <= x1 or y2 <= y1:
        return None
    return int(x1), int(y1), int(x2), int(y2)


def _detect_chunk(
    detector: _FaceDetector,
    frames: List[np.ndarray],
    pads: Tuple[int, int, int, int],
) -> List[Optional[Tuple[int, int, int, int]]]:
    """Detect one face per frame, then apply upstream padding/clamping.

    Large frames are downscaled for the detector only (see ``_DETECT_MAX_SIDE``)
    to keep CPU-only face detection from dominating the run time.
    """
    global _detect_notice_logged
    import cv2  # noqa: PLC0415

    pad_top, pad_bottom, pad_left, pad_right = pads
    height, width = frames[0].shape[:2]
    kx = ky = 1.0

    to_run = frames
    if _DETECT_MAX_SIDE and max(height, width) > _DETECT_MAX_SIDE:
        scale = _DETECT_MAX_SIDE / max(height, width)
        tw = max(1, int(width * scale))
        th = max(1, int(height * scale))
        to_run = [cv2.resize(f, (tw, th)) for f in frames]
        kx, ky = width / tw, height / th
        if not _detect_notice_logged:
            _detect_notice_logged = True
            LOG.info(
                "lipsync: detecting faces on a %dx%d copy of the %dx%d frames "
                "(CPU speed); crops still use the full-resolution frame",
                tw, th, width, height,
            )

    rects = detector.detect_batch(np.asarray(to_run))
    out: List[Optional[Tuple[int, int, int, int]]] = []
    for rect, frame in zip(rects, frames):
        if rect is not None and (kx != 1.0 or ky != 1.0):
            x1, y1, x2, y2 = rect
            rect = (int(x1 * kx), int(y1 * ky), int(x2 * kx), int(y2 * ky))
        out.append(
            _apply_pads(rect, frame.shape[0], frame.shape[1],
                        pad_top, pad_bottom, pad_left, pad_right)
        )
    return out


def _read_sequential(cap, cv2, resize_factor: int, want: int, pos: int) -> Optional[np.ndarray]:
    """Read frame ``want`` from an open capture walking forward from ``pos``."""
    if want != pos:
        cap.set(cv2.CAP_PROP_POS_FRAMES, want)
    return _read_frame(cap, cv2, resize_factor)


def lipsync_fallback(
    video_path: Path,
    audio_path: Path,
    out_path: Path,
    *,
    reason: str = "",
) -> Path:
    """Attach the dubbed audio to the untouched original video.

    Used when Wav2Lip is unavailable or refused the job (e.g. no face).  The
    video stream is copied bit-for-bit, so nothing but the audio changes.
    """
    video_path = Path(video_path)
    audio_path = Path(audio_path)
    out_path = Path(out_path)

    LOG.warning("=" * 60)
    LOG.warning("LIP SYNC SKIPPED - the dubbed audio is muxed over the original")
    LOG.warning("video, so mouth movements will NOT match the new voice.")
    if reason:
        LOG.warning("reason: %s", reason)
    LOG.warning("=" * 60)

    if not video_path.is_file() or video_path.stat().st_size == 0:
        raise RuntimeError(f"lipsync fallback: video not found or empty: {video_path}")
    if not audio_path.is_file() or audio_path.stat().st_size == 0:
        raise RuntimeError(f"lipsync fallback: audio not found or empty: {audio_path}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mux_video_audio(video_path, audio_path, out_path, copy_video=True)
    except CommandError as e:
        LOG.warning(
            "lipsync fallback: cannot copy the video stream (%s); re-encoding instead",
            str(e).splitlines()[0],
        )
        mux_video_audio(video_path, audio_path, out_path, copy_video=False,
                        preset="veryfast", crf=20)

    if not out_path.is_file() or out_path.stat().st_size == 0:
        raise RuntimeError(f"lipsync fallback: no output produced at {out_path}")
    LOG.info("lipsync fallback: wrote %s (no lip sync)", out_path)
    return out_path
