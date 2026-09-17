#!/usr/bin/env python
"""Smoke-test F5-TTS: download the model (once) and synthesise a short line.

This is the single riskiest part of the whole project on a CPU-only box, so it
is worth validating on its own before running the full pipeline.

    python tools/test_f5_smoke.py
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Importing config first sets MPLCONFIGDIR to a writable directory, which
# matplotlib (imported by F5-TTS) requires on this host.
from dublador.config import MODELS_DIR, ensure_dirs, tool_env  # noqa: E402

os.environ.update(tool_env())
ensure_dirs()

from dublador.compat import install_torchaudio_shims  # noqa: E402

shim = install_torchaudio_shims()

from dublador.utils import setup_logging  # noqa: E402

LOG = setup_logging(verbose=False)

import soundfile as sf  # noqa: E402


def main() -> int:
    from f5_tts.api import F5TTS  # noqa: PLC0415

    ref = ROOT / ".python" / "Lib" / "site-packages" / "f5_tts" / "infer" / "examples" / "basic" / "basic_ref_en.wav"
    if not ref.exists():
        print(f"ERROR: bundled reference not found at {ref}")
        return 1

    ref_text = "Some call me nature, others call me mother nature."
    gen_text = "Good evening. The system is finally online, and everything is ready."

    out_dir = ROOT / "work" / "_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_wav = out_dir / "f5_smoke.wav"

    print(f"torchaudio shim installed: {shim}")
    print(f"model cache : {MODELS_DIR / 'hf'}")
    print("loading F5-TTS (downloads ~1.4 GB on first run)...")

    t0 = time.time()
    model = F5TTS(
        model="F5TTS_v1_Base",
        device="cpu",
        hf_cache_dir=str(MODELS_DIR / "hf"),
    )
    load_s = time.time() - t0
    print(f"model loaded in {load_s:.1f}s  (sample rate {model.target_sample_rate} Hz)")

    class Quiet:
        enabled = False

        @staticmethod
        def tqdm(iterable=None, **kw):  # noqa: ANN001
            return iterable if iterable is not None else []

    print("synthesising...")
    t1 = time.time()
    try:
        wav, sr, _spec = model.infer(
            ref_file=str(ref),
            ref_text=ref_text,
            gen_text=gen_text,
            show_info=lambda *a, **k: None,
            progress=Quiet,
            nfe_step=32,
            cfg_strength=2.0,
            speed=1.0,
            fix_duration=None,
            remove_silence=False,
            file_wave=str(out_wav),
            file_spec=None,
            seed=1234,
        )
    except Exception as e:  # noqa: BLE001
        import traceback  # noqa: PLC0415

        traceback.print_exc()
        print(f"\nSYNTHESIS FAILED: {type(e).__name__}: {e}")
        return 1

    wall = time.time() - t1
    duration = len(wav) / float(sr) if sr else 0.0
    info = sf.info(str(out_wav)) if out_wav.exists() else None

    print()
    print("=" * 60)
    print("  F5-TTS SMOKE TEST PASSED")
    print("=" * 60)
    print(f"  output      : {out_wav}")
    print(f"  audio       : {duration:.2f}s @ {sr} Hz")
    if info:
        print(f"  on disk     : {info.frames} frames, {info.channels} ch, {out_wav.stat().st_size / 1024:.0f} KB")
    print(f"  synth time  : {wall:.1f}s")
    print(f"  realtime    : {wall / duration:.1f}x slower than realtime" if duration else "")
    print(f"  load time   : {load_s:.1f}s")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
