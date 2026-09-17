#!/usr/bin/env python
"""Diagnostics: translation backends and speaker separability.

    .python\\python.exe tools\\diag.py [job_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from dublador.speakers import (choose_backend, embed_clip, estimate_f0,  # noqa: E402
                               load_mono)
from dublador.translate import TextTranslator  # noqa: E402


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a / (np.linalg.norm(a) + 1e-9), b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def main() -> int:
    job = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "work" / "test_speech_pt"

    # ---- translation ----------------------------------------------------
    print("=" * 66)
    print("  TRANSLATION")
    print("=" * 66)
    t = TextTranslator("en", "pt", backend="google", workers=1)
    samples = [
        "Good evening. The system is online.",
        "It took longer than you promised.",
        "Everything is ready now.",
    ]
    for s in samples:
        try:
            print(f"  EN: {s}")
            print(f"  PT: {t.translate(s)}")
        except Exception as e:  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}")
    print(f"  stats: {t.stats}")

    # ---- speakers -------------------------------------------------------
    print()
    print("=" * 66)
    print("  SPEAKER SEPARABILITY")
    print("=" * 66)

    tr_path = job / "transcript.json"
    wav_path = job / "audio_16k.wav"
    if not tr_path.exists() or not wav_path.exists():
        print(f"  missing {tr_path} or {wav_path} - run the pipeline first")
        return 1

    data = json.loads(tr_path.read_text(encoding="utf-8"))
    y, sr = load_mono(wav_path)
    print(f"  audio: {len(y) / sr:.2f}s @ {sr} Hz, backend={choose_backend('auto')}")
    print()

    embs = []
    for seg in data["segments"]:
        a, b = int(seg["start"] * sr), int(seg["end"] * sr)
        clip = y[a:b]
        f0 = estimate_f0(clip, sr)
        voiced = f0[f0 > 0]
        pitch = float(np.mean(voiced)) if voiced.size else 0.0
        emb = embed_clip(clip, sr, choose_backend("auto"))
        embs.append(emb)
        text = (seg.get("text") or "").strip()[:44]
        print(f"  seg {seg['id']}: {seg['start']:5.2f}-{seg['end']:5.2f}s  "
              f"pitch {pitch:6.1f} Hz  {text!r}")

    print()
    print("  pairwise cosine similarity (1.0 = identical voice):")
    for i in range(len(embs)):
        for j in range(i + 1, len(embs)):
            if embs[i] is None or embs[j] is None:
                print(f"    {i}-{j}: n/a")
                continue
            sim = cosine(embs[i], embs[j])
            print(f"    {i}-{j}: {sim:+.4f}   (distance {1 - sim:+.4f})")

    print()
    print("  NOTE: the pipeline's default speaker_threshold=0.62 means")
    print("        'same speaker if cosine similarity >= 0.62'.")
    print("        Lower it (e.g. 0.75, or 0.85) to split more aggressively.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
