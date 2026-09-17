"""FFmpeg / FFprobe wrappers.

All external calls go through :func:`run_command`, which redirects child output
into a temp file because this sandbox forbids stdio pipes.
"""
from __future__ import annotations

import math
import os
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from .config import FFMPEG_BIN, FFPROBE_BIN, tool_env
from .utils import LOG, CommandError, CommandResult, ensure_dir, run_command

PathLike = Union[str, Path]


# --------------------------------------------------------------------------
# Info
# --------------------------------------------------------------------------
@dataclass
class MediaInfo:
    path: str
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    video_codec: str = ""
    audio_codec: str = ""
    audio_channels: int = 0
    audio_sample_rate: int = 0
    has_video: bool = False
    has_audio: bool = False
    n_frames: int = 0
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}" if self.has_video else "audio-only"


def _ffmpeg(args: Sequence[PathLike], *, check: bool = True, timeout: Optional[float] = None,
            desc: Optional[str] = None) -> CommandResult:
    cmd = [FFMPEG_BIN, "-hide_banner", "-nostdin", "-y", *[str(a) for a in args]]
    return run_command(cmd, check=check, timeout=timeout, env=tool_env(), desc=desc)


def ffprobe_json(path: PathLike) -> Dict[str, Any]:
    cmd = [
        FFPROBE_BIN, "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ]
    res = run_command(cmd, check=True, env=tool_env(), desc="ffprobe")
    import json  # noqa: PLC0415

    return json.loads(res.output or "{}")


def media_info(path: PathLike) -> MediaInfo:
    info = MediaInfo(path=str(path))
    try:
        data = ffprobe_json(path)
    except Exception as e:  # noqa: BLE001
        LOG.warning("ffprobe failed for %s: %s", path, e)
        return info

    info.raw = data
    fmt = data.get("format", {})
    try:
        info.duration = float(fmt.get("duration") or 0.0)
    except (TypeError, ValueError):
        info.duration = 0.0

    for st in data.get("streams", []):
        kind = st.get("codec_type")
        if kind == "video":
            info.has_video = True
            info.video_codec = st.get("codec_name", "")
            info.width = int(st.get("width") or 0)
            info.height = int(st.get("height") or 0)
            info.n_frames = int(st.get("nb_frames") or 0)
            info.fps = _parse_fps(st.get("avg_frame_rate") or st.get("r_frame_rate") or "0/0")
            if not info.duration:
                info.duration = _to_float(st.get("duration"))
        elif kind == "audio":
            info.has_audio = True
            info.audio_codec = st.get("codec_name", "")
            info.audio_channels = int(st.get("channels") or 0)
            info.audio_sample_rate = int(st.get("sample_rate") or 0)

    if not info.duration:
        for st in data.get("streams", []):
            info.duration = max(info.duration, _to_float(st.get("duration")))
    return info


def _to_float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _parse_fps(rate: str) -> float:
    try:
        num, _, den = str(rate).partition("/")
        n, d = float(num), float(den or 1)
        return n / d if d else 0.0
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def audio_duration(path: PathLike) -> float:
    try:
        data = ffprobe_json(path)
    except Exception:  # noqa: BLE001
        return 0.0
    d = _to_float(data.get("format", {}).get("duration"))
    if d:
        return d
    for st in data.get("streams", []):
        d = max(d, _to_float(st.get("duration")))
    return d


# --------------------------------------------------------------------------
# Audio extraction / conversion
# --------------------------------------------------------------------------
def extract_audio(
    video: PathLike,
    out_wav: PathLike,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    start: Optional[float] = None,
    duration: Optional[float] = None,
    normalize: bool = False,
) -> Path:
    """Extract a clean PCM WAV track (16 kHz mono is what Whisper wants)."""
    ensure_dir(Path(out_wav).parent)
    args: List[Any] = []
    if start is not None:
        args += ["-ss", f"{start:.6f}"]
    args += ["-i", video]
    if duration is not None:
        args += ["-t", f"{duration:.6f}"]
    args += ["-vn", "-map", "0:a:0?", "-ac", str(channels), "-ar", str(sample_rate),
             "-c:a", "pcm_s16le"]
    if normalize:
        args += ["-af", "loudnorm=I=-18:TP=-2:LRA=11"]
    args += [out_wav]
    _ffmpeg(args, desc="extract-audio")
    return Path(out_wav)


def convert_audio(src: PathLike, dst: PathLike, *, sample_rate: int = 24000,
                  channels: int = 1) -> Path:
    ensure_dir(Path(dst).parent)
    _ffmpeg(["-i", src, "-vn", "-ac", str(channels), "-ar", str(sample_rate),
             "-c:a", "pcm_s16le", dst], desc="convert-audio")
    return Path(dst)


def slice_audio(src: PathLike, dst: PathLike, start: float, end: float,
                *, sample_rate: int = 24000, channels: int = 1) -> Path:
    ensure_dir(Path(dst).parent)
    dur = max(0.01, end - start)
    _ffmpeg(["-ss", f"{start:.6f}", "-i", src, "-t", f"{dur:.6f}",
             "-ac", str(channels), "-ar", str(sample_rate),
             "-c:a", "pcm_s16le", dst], desc="slice-audio")
    return Path(dst)


def make_silence(dst: PathLike, duration: float, *, sample_rate: int = 24000,
                 channels: int = 1) -> Path:
    ensure_dir(Path(dst).parent)
    _ffmpeg(["-f", "lavfi", "-i",
             f"anullsrc=channel_layout={'mono' if channels == 1 else 'stereo'}:sample_rate={sample_rate}",
             "-t", f"{max(0.01, duration):.6f}", "-c:a", "pcm_s16le", dst], desc="silence")
    return Path(dst)


def concat_audio(parts: Iterable[PathLike], dst: PathLike, *, sample_rate: int = 24000,
                 channels: int = 1) -> Path:
    """Sample-accurate concatenation, normalising every part first."""
    parts = [Path(p) for p in parts]
    if not parts:
        raise ValueError("concat_audio needs at least one part")
    ensure_dir(Path(dst).parent)
    if len(parts) == 1:
        convert_audio(parts[0], dst, sample_rate=sample_rate, channels=channels)
        return Path(dst)

    list_file = Path(dst).with_suffix(".concat.txt")
    lines = []
    for p in parts:
        safe = str(p.resolve()).replace("'", "'\\''")
        lines.append(f"file '{safe}'")
    list_file.write_text("\n".join(lines) + "\n", encoding="utf-8")

    _ffmpeg(["-f", "concat", "-safe", "0", "-i", list_file,
             "-ac", str(channels), "-ar", str(sample_rate),
             "-c:a", "pcm_s16le", dst], desc="concat-audio")
    try:
        list_file.unlink()
    except OSError:
        pass
    return Path(dst)


# --------------------------------------------------------------------------
# Time fitting
# --------------------------------------------------------------------------
_HAS_RUBBERBAND: Optional[bool] = None


def has_rubberband() -> bool:
    """This gyan.dev build ships librubberband, which beats atempo for speech."""
    global _HAS_RUBBERBAND
    if _HAS_RUBBERBAND is None:
        try:
            res = _ffmpeg(["-hide_banner", "-filters"], check=False, desc="filters")
            _HAS_RUBBERBAND = "rubberband" in res.output
        except Exception:  # noqa: BLE001
            _HAS_RUBBERBAND = False
    return bool(_HAS_RUBBERBAND)


def _atempo_chain(factor: float) -> str:
    """atempo accepts 0.5..2.0 per instance (wider in recent builds)."""
    factor = max(0.05, min(20.0, factor))
    parts: List[str] = []
    remaining = factor
    while remaining > 2.0:
        parts.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.6f}")
    return ",".join(parts)


def time_stretch(src: PathLike, dst: PathLike, factor: float, *, sample_rate: int = 24000,
                 channels: int = 1, high_quality: bool = True) -> Path:
    """Change duration by ``factor`` (>1 = faster) preserving pitch."""
    ensure_dir(Path(dst).parent)
    if abs(factor - 1.0) < 0.005:
        return convert_audio(src, dst, sample_rate=sample_rate, channels=channels)

    if high_quality and has_rubberband():
        # rubberband tempo: pitch preserved
        flt = f"rubberband=tempo={max(0.05, min(20.0, factor)):.6f}"
    else:
        flt = _atempo_chain(factor)
    _ffmpeg(["-i", src, "-vn", "-af", flt, "-ac", str(channels),
             "-ar", str(sample_rate), "-c:a", "pcm_s16le", dst], desc="time-stretch")
    return Path(dst)


def fit_to_slot(src: PathLike, dst: PathLike, target_duration: float, *,
                sample_rate: int = 24000, channels: int = 1,
                max_speed: float = 1.40, min_speed: float = 0.72,
                pad: bool = True) -> Tuple[Path, Dict[str, Any]]:
    """Force ``src`` to occupy exactly ``target_duration`` seconds.

    Strategy
    --------
    * If the clip is longer than the slot, speed it up -- but never beyond
      ``max_speed`` (past that, speech becomes unintelligible).  If it still
      overflows it is truncated.
    * If it is shorter, slow it down only slightly (``min_speed``) and pad the
      remainder with silence so the following line keeps its timing.
    """
    meta: Dict[str, Any] = {"original": audio_duration(src), "target": target_duration}
    cur = meta["original"]
    if cur <= 0.0:
        make_silence(dst, target_duration, sample_rate=sample_rate, channels=channels)
        meta.update({"action": "silence-empty", "final": target_duration})
        return Path(dst), meta

    factor = 1.0
    ratio = cur / target_duration

    if ratio > 1.0:
        factor = min(ratio, max_speed)
    elif ratio < 1.0:
        factor = max(ratio, min_speed)

    work = Path(dst).with_name(Path(dst).stem + ".stretch.wav")
    if abs(factor - 1.0) > 0.005:
        time_stretch(src, work, factor, sample_rate=sample_rate, channels=channels)
    else:
        convert_audio(src, work, sample_rate=sample_rate, channels=channels)

    cur = audio_duration(work)
    meta["factor"] = round(factor, 3)
    meta["after_stretch"] = cur

    if pad and cur < target_duration - 0.01:
        tail = Path(dst).with_name(Path(dst).stem + ".pad.wav")
        make_silence(tail, target_duration - cur, sample_rate=sample_rate, channels=channels)
        concat_audio([work, tail], dst, sample_rate=sample_rate, channels=channels)
        for f in (tail,):
            try:
                f.unlink()
            except OSError:
                pass
        meta["action"] = "stretched+padded"
    elif cur > target_duration + 0.01:
        slice_audio(work, dst, 0.0, target_duration, sample_rate=sample_rate, channels=channels)
        meta["action"] = "stretched+trimmed"
    else:
        convert_audio(work, dst, sample_rate=sample_rate, channels=channels)
        meta["action"] = "as-is"

    try:
        work.unlink()
    except OSError:
        pass
    meta["final"] = audio_duration(dst)
    return Path(dst), meta


# --------------------------------------------------------------------------
# Mixing / muxing
# --------------------------------------------------------------------------
def mix_audio(
    primary: PathLike,
    secondary: Optional[PathLike],
    dst: PathLike,
    *,
    primary_db: float = 0.0,
    secondary_db: float = -12.0,
    duck: bool = False,
    sample_rate: int = 24000,
    channels: int = 1,
    duration: Optional[float] = None,
) -> Path:
    """Mix a dubbed voice track over an optional background bed."""
    ensure_dir(Path(dst).parent)
    if secondary is None or not Path(secondary).exists():
        args: List[Any] = ["-i", primary]
        if duration:
            args += ["-t", f"{duration:.3f}"]
        args += ["-af", f"volume={primary_db:.2f}dB", "-ac", str(channels),
                 "-ar", str(sample_rate), "-c:a", "pcm_s16le", dst]
        _ffmpeg(args, desc="mix-voice-only")
        return Path(dst)

    if duck:
        # Side-chain compress the background against the voice for clarity.
        flt = (
            f"[0:a]volume={primary_db:.2f}dB,asplit=2[v1][v2];"
            f"[1:a]volume={secondary_db:.2f}dB[bg];"
            f"[bg][v2]sidechaincompress=threshold=0.05:ratio=8:attack=20:release=400[bgd];"
            f"[v1][bgd]amix=inputs=2:duration=longest:normalize=0[mix]"
        )
        args = ["-i", primary, "-i", secondary, "-filter_complex", flt,
                "-map", "[mix]"]
    else:
        args = ["-i", primary, "-i", secondary, "-filter_complex",
                f"[0:a]volume={primary_db:.2f}dB[v];[1:a]volume={secondary_db:.2f}dB[b];"
                f"[v][b]amix=inputs=2:duration=longest:normalize=0[mix]",
                "-map", "[mix]"]

    if duration:
        args += ["-t", f"{duration:.3f}"]
    args += ["-ac", str(channels), "-ar", str(sample_rate), "-c:a", "pcm_s16le", dst]
    _ffmpeg(args, desc="mix-audio")
    return Path(dst)


def mux_video_audio(
    video: PathLike,
    audio: PathLike,
    dst: PathLike,
    *,
    copy_video: bool = True,
    video_codec: str = "libx264",
    crf: int = 18,
    preset: str = "medium",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    shortest: bool = True,
    faststart: bool = True,
    extra_args: Optional[Sequence[PathLike]] = None,
) -> Path:
    """Attach ``audio`` to ``video`` replacing every original audio stream."""
    ensure_dir(Path(dst).parent)
    args: List[Any] = ["-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0"]
    if copy_video:
        args += ["-c:v", "copy"]
    else:
        args += ["-c:v", video_codec, "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p"]
    args += ["-c:a", audio_codec, "-b:a", audio_bitrate, "-ac", "2", "-ar", "48000"]
    if shortest:
        args += ["-shortest"]
    if faststart and audio_codec != "copy":
        args += ["-movflags", "+faststart"]
    if extra_args:
        args += list(extra_args)
    args += [dst]
    _ffmpeg(args, desc="mux")
    return Path(dst)


def replace_audio_track(video: PathLike, dst: PathLike, *,
                        video_codec: str = "libx264", crf: int = 18,
                        preset: str = "medium") -> Path:
    """Produce a video with no audio stream at all."""
    ensure_dir(Path(dst).parent)
    _ffmpeg(["-i", video, "-map", "0:v:0", "-an", "-c:v", video_codec,
             "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p", dst],
            desc="strip-audio")
    return Path(dst)


def extract_frames(video: PathLike, out_dir: PathLike, *, fps: Optional[float] = None) -> Path:
    ensure_dir(out_dir)
    args: List[Any] = ["-i", video]
    if fps:
        args += ["-vf", f"fps={fps}"]
    args += [str(Path(out_dir) / "frame_%08d.png")]
    _ffmpeg(args, desc="extract-frames")
    return Path(out_dir)


def encode_frames_to_video(frames_dir: PathLike, audio: PathLike, dst: PathLike, *,
                           fps: float = 25.0, crf: int = 18) -> Path:
    ensure_dir(Path(dst).parent)
    _ffmpeg(["-framerate", f"{fps}", "-i", str(Path(frames_dir) / "frame_%08d.png"),
             "-i", audio, "-map", "0:v:0", "-map", "1:a:0",
             "-c:v", "libx264", "-crf", str(crf), "-preset", "medium",
             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
             "-shortest", dst], desc="encode-frames")
    return Path(dst)


def check_ffmpeg() -> bool:
    if not Path(FFMPEG_BIN).exists():
        LOG.error("ffmpeg not found at %s", FFMPEG_BIN)
        return False
    try:
        res = _ffmpeg(["-version"], check=False)
        return res.returncode == 0
    except Exception:  # noqa: BLE001
        return False
