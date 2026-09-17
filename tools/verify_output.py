#!/usr/bin/env python
"""Verify a dubbed video: is there real, intelligible speech in the target language?

Round-trips the output through Whisper.  If the dub were silent, truncated or
still in the source language, this would show it.

    .python\\python.exe tools\\verify_output.py output\\x_dubbed.mp4 --expect pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from dublador import media  # noqa: E402


def dbfs(peak: float) -> str:
    if peak <= 1e-9:
        return "-inf"
    return f"{20 * np.log10(peak):.1f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--expect", default=None, help="expected language code")
    ap.add_argument("--work", default=str(ROOT / "work" / "_verify"))
    args = ap.parse_args()

    video = Path(args.video)
    if not video.is_absolute():
        video = (ROOT / video).resolve()
    if not video.exists():
        print(f"not found: {video}")
        return 1

    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    print("=" * 66)
    print("  OUTPUT VERIFICATION")
    print("=" * 66)

    info = media.media_info(video)
    print(f"  file      : {video}")
    print(f"  size      : {video.stat().st_size / 1048576:.2f} MB")
    print(f"  duration  : {info.duration:.2f}s   {info.resolution} @ {info.fps:.2f} fps")
    print(f"  video     : {info.video_codec}")
    print(f"  audio     : {info.audio_codec}, {info.audio_channels} ch, {info.audio_sample_rate} Hz")

    if not info.has_audio:
        print("\n  FAIL: the output has NO audio stream")
        return 1

    # ---- level analysis -------------------------------------------------
    wav = work / "verify_16k.wav"
    media.extract_audio(video, wav, sample_rate=16000, channels=1)

    import soundfile as sf  # noqa: PLC0415

    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    mono = data[:, 0]
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0

    frame = max(1, int(0.02 * sr))
    n = mono.size // frame
    frame_rms = np.sqrt(np.mean(mono[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12) if n else np.zeros(1)
    speech_frames = int(np.sum(frame_rms > max(rms * 0.25, 1e-4)))
    speech_ratio = speech_frames / max(1, len(frame_rms))

    print()
    print(f"  peak      : {peak:.4f}  ({dbfs(peak)} dBFS)")
    print(f"  rms       : {rms:.4f}  ({dbfs(rms)} dBFS)")
    print(f"  voiced    : {speech_ratio * 100:.1f}% of frames above the noise floor")

    ok = True
    if peak < 0.01:
        print("  FAIL: the dub is effectively SILENT")
        ok = False
    elif speech_ratio < 0.10:
        print("  WARN: very little voiced audio - check fit_to_slot results")
    else:
        print("  OK: the output contains real, non-silent audio")

    # ---- intelligibility -------------------------------------------------
    print()
    print("  transcribing the dubbed audio back (round-trip check)...")
    from dublador.transcribe import transcribe  # noqa: PLC0415

    # Auto-detect first, purely informational: a voice cloned from a source
    # language reference keeps that accent, so the detector can legitimately
    # guess the source language even when the words are clearly the target one.
    auto = transcribe(wav, model_size="small", device="cpu", compute_type="int8",
                      beam_size=5, vad_filter=True)
    print(f"  auto-detected language : {auto.language} ({auto.language_probability:.0%})")
    if args.expect and not (auto.language or "").lower().startswith(args.expect.lower()):
        print("    (low confidence is expected: the clone keeps the source accent)")

    # Force the expected language for the actual content comparison.
    tr = auto if not args.expect else transcribe(
        wav, model_size="small", language=args.expect, device="cpu",
        compute_type="int8", beam_size=5, vad_filter=True,
    )
    print(f"  segments               : {len(tr.segments)}")
    print()
    for seg in tr.segments:
        print(f"    [{seg.start:6.2f} - {seg.end:6.2f}]  {seg.text}")

    # Compare against what the pipeline actually generated.
    expected_text = _expected_translation(video)
    if expected_text:
        heard = " ".join(s.text for s in tr.segments)
        ratio = _word_overlap(heard, expected_text)
        print()
        print(f"  expected (pipeline translation): {expected_text[:110]}")
        print(f"  word overlap with what we heard: {ratio * 100:.0f}%")
        if ratio >= 0.35:
            print("  OK: the dub carries the intended translated speech")
        else:
            print("  FAIL: the dubbed speech does not resemble the translation")
            ok = False
    elif args.expect:
        print()
        print("  (no transcript_translated.json found to compare against)")

    print()
    print("  RESULT:", "PASS" if ok else "FAIL")
    print()
    return 0 if ok else 1


def _expected_translation(video: Path) -> str:
    """Find the pipeline's own translated transcript next to the output."""
    stem = video.stem
    for cand in (ROOT / "work").glob("*/transcript_translated.json"):
        if cand.parent.name.startswith(stem.split("_")[0]):
            try:
                data = json.loads(cand.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            parts = [s.get("translation") or s.get("text") or ""
                     for s in data.get("segments", [])]
            joined = " ".join(p for p in parts if p).strip()
            if joined:
                return joined
    return ""


def _word_overlap(a: str, b: str) -> float:
    """Fraction of the expected words that appear in the heard text."""
    import re  # noqa: PLC0415
    import unicodedata  # noqa: PLC0415

    def norm(text: str) -> list:
        text = unicodedata.normalize("NFKD", text.lower())
        text = "".join(c for c in text if not unicodedata.combining(c))
        return [w for w in re.findall(r"[a-z0-9]+", text) if len(w) > 1]

    want, got = norm(b), set(norm(a))
    if not want:
        return 0.0
    return sum(1 for w in want if w in got) / len(want)


if __name__ == "__main__":
    sys.exit(main())
