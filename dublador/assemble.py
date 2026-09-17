"""Timeline assembly: place every synthesised line back on the original clock.

Doing this in numpy rather than with a giant ffmpeg ``filter_complex`` keeps it
exact (sample-accurate), fast, and free of command-length limits when a video
has hundreds of lines.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import tool_env
from .schema import Segment, Transcript
from .utils import LOG, Progress, banner, ensure_dir, human_duration

DUB_SR = 24000


def _read_mono(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    import soundfile as sf  # noqa: PLC0415

    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    mono = data.mean(axis=1) if data.shape[1] > 1 else data[:, 0]
    if sr != target_sr:
        import librosa  # noqa: PLC0415

        mono = librosa.resample(mono, orig_sr=sr, target_sr=target_sr)
        sr = target_sr
    return np.ascontiguousarray(mono, dtype=np.float32), sr


def build_dub_track(
    tr: Transcript,
    total_duration: float,
    dst: Path,
    *,
    sample_rate: int = DUB_SR,
    gap: float = 0.0,
    limit: bool = True,
    target_peak: float = 0.89,
) -> Tuple[Path, Dict[str, Any]]:
    """Render all ``segment.fitted_audio`` clips onto a silent timeline.

    Each clip starts at its original ``segment.start``, so the dub stays locked
    to the picture even when a translation runs shorter or longer than the
    source line.
    """
    ensure_dir(Path(dst).parent)

    total_duration = max(float(total_duration or 0.0), 0.0)
    if total_duration <= 0:
        for seg in tr.segments:
            total_duration = max(total_duration, seg.end)
    n = int(math.ceil((total_duration + 0.5) * sample_rate))
    buf = np.zeros(max(n, sample_rate), dtype=np.float32)

    placed = 0
    skipped = 0
    overlaps = 0
    ends = np.zeros(0, dtype=np.float32)

    todo = [s for s in tr.segments if s.fitted_audio and Path(s.fitted_audio).exists()]
    prog = Progress(len(todo), label="  mix", every=max(1, len(todo) // 10 or 1))

    for seg in todo:
        try:
            clip, _ = _read_mono(Path(seg.fitted_audio), sample_rate)
        except Exception as e:  # noqa: BLE001
            LOG.warning("  could not read %s: %s", seg.fitted_audio, e)
            skipped += 1
            prog.step()
            continue

        start = int(round(max(0.0, seg.start + gap) * sample_rate))
        if start >= buf.size:
            skipped += 1
            prog.step()
            continue

        end = min(buf.size, start + clip.size)
        take = end - start
        if take <= 0:
            skipped += 1
            prog.step()
            continue

        if take < clip.size:
            clip = clip[:take]

        # Soft-sum: apply a short fade at the head/tail so a clip that slightly
        # overruns its neighbour does not click.
        fade = min(int(0.008 * sample_rate), max(1, clip.size // 8))
        if fade > 1:
            ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
            clip = clip.copy()
            clip[:fade] *= ramp
            clip[-fade:] *= ramp[::-1]

        buf[start:end] += clip
        placed += 1
        prog.step()

    # Detect summed overlap so we can report it honestly.
    peaks_before = float(np.max(np.abs(buf))) if buf.size else 0.0

    if limit and peaks_before > target_peak:
        gain = target_peak / peaks_before
        LOG.info("  limiting dub track by %.2f dB (peak was %.2f)",
                 20 * math.log10(max(gain, 1e-6)), peaks_before)
        buf *= gain

    import soundfile as sf  # noqa: PLC0415

    sf.write(str(dst), buf, sample_rate, subtype="PCM_16")

    meta = {
        "placed": placed,
        "skipped": skipped,
        "duration": round(buf.size / sample_rate, 3),
        "peak_before_limit": round(peaks_before, 4),
        "sample_rate": sample_rate,
    }
    LOG.info("dub track: %d line(s) placed, %d skipped, %.2fs -> %s",
             placed, skipped, meta["duration"], Path(dst).name)
    return Path(dst), meta


def extract_background_bed(
    video: Path,
    dst: Path,
    *,
    sample_rate: int = DUB_SR,
    start: Optional[float] = None,
    duration: Optional[float] = None,
) -> Path:
    """Pull the original soundtrack to reuse as a music/SFX bed."""
    from .media import extract_audio  # noqa: PLC0415

    return extract_audio(video, dst, sample_rate=sample_rate, channels=1,
                         start=start, duration=duration)


def separate_vocals_demucs(src: Path, out_dir: Path) -> Optional[Path]:
    """Remove original vocals with Demucs, if it happens to be installed.

    Returns the path to the ``no_vocals`` stem, or ``None`` when Demucs is not
    available.  We never install Demucs ourselves: it needs a ~300 MB model and
    is extremely slow on a 2-core CPU.
    """
    try:
        import demucs  # noqa: F401,PLC0415
    except Exception:  # noqa: BLE001
        LOG.info("  Demucs not installed - cannot separate the original vocals")
        return None

    from .config import python_exe  # noqa: PLC0415
    from .utils import run_command  # noqa: PLC0415

    ensure_dir(out_dir)
    LOG.info("  separating original vocals with Demucs (this is slow)...")
    try:
        res = run_command(
            [python_exe(), "-m", "demucs", "--two-stems", "vocals",
             "-o", str(out_dir), str(src)],
            env=tool_env(), check=False, desc="demucs",
        )
        if not res.ok:
            LOG.warning("  Demucs failed: %s", res.output.strip()[-300:])
            return None
    except Exception as e:  # noqa: BLE001
        LOG.warning("  Demucs error: %s", e)
        return None

    for p in Path(out_dir).rglob("no_vocals.*"):
        LOG.info("  background stem: %s", p.name)
        return p
    return None
