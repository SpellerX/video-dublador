"""Host compatibility shims, applied before any heavy third-party import.

Problem
-------
``torchaudio`` >= 2.9 removed its native audio I/O and now delegates
``torchaudio.load`` / ``torchaudio.save`` to **TorchCodec**, which is not
installed here (it also needs FFmpeg *shared* libraries, while our bundled
ffmpeg is a static build).  F5-TTS calls ``torchaudio.load`` in
``utils_infer.preprocess_ref_audio_text``, so without a fix every synthesis
would raise ``ImportError: TorchCodec is required``.

Fix
---
If TorchCodec is unavailable, replace ``torchaudio.load`` / ``save`` / ``info``
with soundfile-backed equivalents that keep the exact torchaudio signature and
return ``(Tensor[channels, frames], sample_rate)``.  ``soundfile`` is already a
dependency and reads/writes wav, flac and ogg natively.

Call :func:`install_torchaudio_shims` once, as early as possible.
"""
from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional, Tuple

from .utils import LOG

_APPLIED = False


def _torchcodec_available() -> bool:
    try:
        import torchcodec  # noqa: F401,PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


def _native_io_works() -> bool:
    """Ask torchaudio itself whether load() can actually run."""
    try:
        import torchaudio  # noqa: PLC0415
        from torchaudio import _torchcodec  # noqa: F401,PLC0415
    except ImportError:
        # Older torchaudio: real native implementation, nothing to do.
        return True
    except Exception:  # noqa: BLE001
        return False
    return _torchcodec_available()


def install_torchaudio_shims(force: bool = False) -> bool:
    """Patch torchaudio audio I/O. Returns True if a shim was installed."""
    global _APPLIED
    if _APPLIED and not force:
        return False

    try:
        import soundfile as sf  # noqa: PLC0415
        import torch  # noqa: PLC0415
        import torchaudio  # noqa: PLC0415
    except ImportError as e:
        LOG.debug("torchaudio shim skipped: %s", e)
        return False

    if not force and _native_io_works():
        _APPLIED = True
        return False

    def load(
        filepath,
        frame_offset: int = 0,
        num_frames: int = -1,
        normalize: bool = True,
        channels_first: bool = True,
        **kwargs: Any,
    ) -> Tuple["torch.Tensor", int]:
        data, sample_rate = sf.read(
            str(filepath),
            start=max(0, int(frame_offset)),
            frames=-1 if num_frames in (-1, None) else int(num_frames),
            dtype="float32" if normalize else "int16",
            always_2d=True,
        )
        tensor = torch.from_numpy(data)
        if channels_first:
            tensor = tensor.transpose(0, 1)
        if not normalize:
            tensor = tensor.float()
        return tensor.contiguous(), int(sample_rate)

    def save(
        filepath,
        src,
        sample_rate: int,
        channels_first: bool = True,
        format: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        import numpy as np  # noqa: PLC0415

        arr = src.detach().cpu() if hasattr(src, "detach") else torch.as_tensor(src)
        if channels_first:
            arr = arr.transpose(0, 1)
        data = arr.numpy()
        if data.dtype.kind == "f":
            data = np.clip(data, -1.0, 1.0)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(filepath), data, int(sample_rate), format=format or "WAV")

    def info(filepath, **kwargs: Any):
        obj = sf.info(str(filepath))
        from collections import namedtuple  # noqa: PLC0415

        AudioMetaData = namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
            defaults=[None, None, None, None, None],
        )
        try:
            subtype = (obj.subtype or "").upper()
            bits = int("".join(c for c in subtype if c.isdigit()) or 16)
        except Exception:  # noqa: BLE001
            bits = 16
        return AudioMetaData(obj.samplerate, obj.frames, obj.channels, bits, obj.subtype)

    torchaudio.load = load
    torchaudio.save = save
    torchaudio.info = info

    # Whatever backend registry exists must not be consulted again.
    try:
        torchaudio.set_audio_backend("soundfile")
    except Exception:  # noqa: BLE001
        pass

    _APPLIED = True
    LOG.debug("installed soundfile-backed torchaudio.load/save/info shims")
    return True


def torchaudio_shim_needed() -> bool:
    return not _native_io_works()
