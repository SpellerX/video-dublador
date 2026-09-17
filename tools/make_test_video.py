#!/usr/bin/env python
"""Build a test video containing real, multi-speaker speech.

This is a *test fixture generator*, not part of the dubbing pipeline.  It uses
F5-TTS itself to synthesise a short two-actor dialogue from the reference voices
bundled with the F5-TTS package, then muxes the result onto a synthetic picture
track.  Feeding the dubber its own output is circular as *content*, but it is a
perfectly good way to obtain speech audio with known ground truth, and it
doubles as an end-to-end check that synthesis works.

    .python\\python.exe tools\\make_test_video.py
    .python\\python.exe tools\\make_test_video.py --nfe-step 16 --out input\\demo.mp4
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dublador.config import MODELS_DIR, INPUT_DIR, ensure_dirs, tool_env  # noqa: E402

os.environ.update(tool_env())
ensure_dirs()

from dublador.compat import install_torchaudio_shims  # noqa: E402

install_torchaudio_shims()

from dublador import media  # noqa: E402
from dublador.utils import LOG, human_duration, setup_logging  # noqa: E402

EXAMPLES = Path(sys.executable).parent / "Lib" / "site-packages" / "f5_tts" / "infer" / "examples"

REF_EN = EXAMPLES / "basic" / "basic_ref_en.wav"
REF_EN_TEXT = "Some call me nature, others call me mother nature."

#: Two actors. BRUNO's reference is derived from the same clip by shifting the
#: formants down, which yields an acoustically distinct second voice without
#: needing another download.
DIALOGUE: List[Dict[str, str]] = [
    {"who": "ANNA",  "text": "Good evening. The system is finally online."},
    {"who": "BRUNO", "text": "I told you it would take longer than you promised."},
    {"who": "ANNA",  "text": "Everything is ready for the demonstration."},
    {"who": "BRUNO", "text": "Then let us begin, before the storm arrives."},
]

#: Synthesis on a CPU-only host costs ~100x realtime, so a smoke-test fixture
#: should stay small.  This one is ~7 s of speech.
SHORT_DIALOGUE: List[Dict[str, str]] = [
    {"who": "ANNA",  "text": "Good evening. The system is online."},
    {"who": "BRUNO", "text": "It took longer than you promised."},
    {"who": "ANNA",  "text": "Everything is ready now."},
]

GAP = 0.55


def derive_low_voice(src: Path, dst: Path, pitch: float = 0.78) -> Path:
    """Shift a reference clip's pitch down to create a second speaker."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    rate = int(22050 * pitch)
    tempo = round(1.0 / pitch, 6)
    media._ffmpeg(  # noqa: SLF001 - intentional reuse of the ffmpeg wrapper
        ["-i", src, "-af", f"asetrate={rate},aresample=22050,atempo={tempo}",
         "-ac", "1", "-c:a", "pcm_s16le", dst],
        desc="derive-voice",
    )
    LOG.info("  derived second voice -> %s", dst.name)
    return dst


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(INPUT_DIR / "test_speech.mp4"))
    ap.add_argument("--work", default=str(ROOT / "work" / "_fixture"))
    ap.add_argument("--nfe-step", type=int, default=24)
    ap.add_argument("--f5-model", default="F5TTS_v1_Base")
    ap.add_argument("--short", action="store_true",
                    help="use a 3-line (~7s) dialogue - far faster on CPU")
    ap.add_argument("--keep-parts", action="store_true")
    args = ap.parse_args(argv)

    setup_logging(False)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)

    if not REF_EN.exists():
        LOG.error("bundled reference not found: %s", REF_EN)
        return 1

    # ---- references ------------------------------------------------------
    LOG.info("preparing voice references")
    ref_bruno = derive_low_voice(REF_EN, work / "ref_bruno.wav")
    voices = {
        "ANNA": (REF_EN, REF_EN_TEXT),
        "BRUNO": (ref_bruno, REF_EN_TEXT),
    }

    # ---- synthesise the dialogue ----------------------------------------
    from f5_tts.api import F5TTS  # noqa: PLC0415

    LOG.info("loading F5-TTS '%s' (downloads ~1.4 GB on first run)", args.f5_model)
    t0 = time.time()
    cloner = F5TTS(model=args.f5_model, device="cpu", hf_cache_dir=str(MODELS_DIR / "hf"))
    LOG.info("  loaded in %.1fs", time.time() - t0)

    class Quiet:
        enabled = False

        @staticmethod
        def tqdm(iterable=None, **kw):  # noqa: ANN001
            return iterable if iterable is not None else []

    dialogue = SHORT_DIALOGUE if args.short else DIALOGUE
    LOG.info("synthesising %d dialogue line(s) for %d voice(s)",
             len(dialogue), len({d["who"] for d in dialogue}))

    parts: List[Path] = []
    t0 = time.time()
    for i, line in enumerate(dialogue, 1):
        ref, ref_text = voices[line["who"]]
        out = work / f"line{i:02d}_{line['who']}.wav"
        LOG.info("[%d/%d] %s: %s", i, len(dialogue), line["who"], line["text"])
        ts = time.time()
        cloner.infer(
            ref_file=str(ref),
            ref_text=ref_text,
            gen_text=line["text"],
            show_info=lambda *a, **k: None,
            progress=Quiet,
            nfe_step=args.nfe_step,
            cfg_strength=2.0,
            speed=1.0,
            remove_silence=False,
            file_wave=str(out),
            file_spec=None,
            seed=1000 + i,
        )
        dur = media.audio_duration(out)
        LOG.info("      -> %.2fs of audio in %.1fs (%.1fx realtime)",
                 dur, time.time() - ts, (time.time() - ts) / dur if dur else 0.0)

        parts.append(out)
        gap = media.make_silence(work / f"gap{i:02d}.wav", GAP, sample_rate=24000)
        parts.append(gap)

    LOG.info("total synthesis time: %s", human_duration(time.time() - t0))

    # ---- concatenate -----------------------------------------------------
    speech = work / "speech.wav"
    media.concat_audio(parts, speech, sample_rate=24000, channels=1)
    duration = media.audio_duration(speech)
    LOG.info("dialogue track: %.2fs", duration)

    # ---- picture + mux ---------------------------------------------------
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    LOG.info("building %s", out_path)

    media._ffmpeg(  # noqa: SLF001
        [
            "-f", "lavfi", "-i", "color=c=0x1b2430:s=640x360:r=25:d=600",
            "-i", speech,
            "-vf",
            ("drawtext=text='VIDEO DUBBER TEST FIXTURE':fontcolor=white:fontsize=22:"
             "x=(w-text_w)/2:y=(h-text_h)/2-30,"
             "drawtext=text='two synthetic speakers, English dialogue':"
             "fontcolor=0x9fb3c8:fontsize=15:x=(w-text_w)/2:y=(h-text_h)/2+10"),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "128k", "-shortest", "-movflags", "+faststart",
            out_path,
        ],
        desc="build-fixture",
    )

    info = media.media_info(out_path)
    LOG.info("")
    LOG.info("  fixture : %s", out_path)
    LOG.info("  duration: %.2fs   %s   %.2f MB", info.duration, info.resolution,
             out_path.stat().st_size / 1048576)
    LOG.info("")
    LOG.info("  now try:  .\\run.ps1 %s --target pt", out_path.name)
    LOG.info("")

    if not args.keep_parts:
        for p in parts:
            try:
                p.unlink()
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
