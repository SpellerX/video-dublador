#!/usr/bin/env python
"""Reclaim space by removing site-packages this project never executes.

Some of the heaviest dependencies were pulled in by code paths the dubber never
runs.  ``f5_tts.model.__init__`` imports ``Trainer`` (training only), which drags
in ``wandb`` and ``datasets`` -> ``pyarrow`` + ``pandas``; ``cached_path``
advertises S3/GCS support; ``sympy`` and ``numba`` come along for the ride.

Removing a package that is genuinely needed would silently break dubbing, so
this tool never trusts a hard-coded list.  For each group it:

  1. renames the package directories aside (``<name>.off``),
  2. runs a **separate interpreter** that exercises the real import and code
     paths the pipeline uses,
  3. restores the group if the probe fails, or keeps it removed if it passes.

Nothing is deleted until a group has passed, so the operation is reversible at
every step.

    .python\\python.exe tools\\trim_env.py            # dry run: test only
    .python\\python.exe tools\\trim_env.py --apply    # remove what passes
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SITE = ROOT / ".python" / "Lib" / "site-packages"

#: Groups are removed together because they are useless apart.
GROUPS: Dict[str, List[str]] = {
    "numba":     ["numba", "llvmlite"],
    "sympy":     ["sympy"],
    "cloud-sdks": ["boto3", "botocore", "s3transfer", "jmespath",
                   "google", "google_auth", "google_auth_oauthlib",
                   "googleapiclient", "google_cloud_storage", "google_crc32c",
                   "google_resumable_media", "googleapis_common_protos"],
    "training-only": ["wandb", "datasets", "pyarrow", "pandas", "dill",
                      "multiprocess", "xxhash"],
}

#: Exercises everything the pipeline actually touches.  Deliberately broad: a
#: missed import here would turn into a broken dub later.
PROBE = r'''
import sys, os, traceback
sys.path.insert(0, r"{root}")
os.environ.update(__import__("dublador.config", fromlist=["x"]).tool_env())

failures = []
def step(name, fn):
    try:
        fn()
        print("  ok   " + name)
    except Exception as e:
        failures.append((name, repr(e)))
        print("  FAIL " + name + " -> " + repr(e))

def _dublador():
    from dublador import (assemble, cli, compat, config, estimate, gui,  # noqa
                          hardware, media, pipeline, schema, speakers,
                          transcribe, translate, tts, utils)

def _torch():
    import torch
    (torch.randn(64, 64) @ torch.randn(64, 64)).sum().item()

def _librosa_real():
    # exactly the calls the voice-analysis stage makes
    import numpy as np, librosa
    y = np.random.randn(16000).astype("float32") * 0.05
    librosa.resample(y, orig_sr=16000, target_sr=24000)
    librosa.effects.trim(y, top_db=32)
    m = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=20, n_fft=1024, hop_length=256)
    librosa.feature.delta(m)
    librosa.feature.spectral_centroid(y=y, sr=16000, n_fft=1024, hop_length=256)
    librosa.feature.spectral_rolloff(y=y, sr=16000, n_fft=1024, hop_length=256)
    f0 = librosa.yin(y, fmin=60, fmax=400, sr=16000, frame_length=1024, hop_length=256)
    assert f0.size > 0
    from librosa.filters import mel
    mel(sr=24000, n_fft=1024, n_mels=100)

def _speakers_real():
    import numpy as np
    from dublador.speakers import embed_clip, cluster_auto
    y = (np.random.randn(16000) * 0.05).astype("float32")
    e = embed_clip(y, 16000, "mfcc")
    assert e is not None and e.size > 10
    mat = np.random.randn(6, e.size)
    cluster_auto(mat / np.linalg.norm(mat, axis=1, keepdims=True))

def _f5_import():
    from dublador.compat import install_torchaudio_shims
    install_torchaudio_shims()
    import f5_tts.api          # trains nothing, but imports the whole tree
    import f5_tts.model        # the Trainer import lives here
    import f5_tts.infer.utils_infer

def _whisper():
    import faster_whisper
    from faster_whisper import WhisperModel  # noqa

def _lipsync():
    import cv2
    from dublador import lipsync
    assert lipsync.lipsync_available()

def _translate():
    from dublador.translate import TextTranslator  # noqa

def _hardware():
    from dublador.hardware import detect, recommend
    hw = detect(deep=True)
    recommend(hw)

step("dublador package", _dublador)
step("torch matmul", _torch)
step("librosa real calls", _librosa_real)
step("speaker embeddings", _speakers_real)
step("f5_tts import tree", _f5_import)
step("faster-whisper", _whisper)
step("wav2lip / opencv", _lipsync)
step("translation", _translate)
step("hardware detect", _hardware)

print("PROBE_RESULT=" + ("PASS" if not failures else "FAIL"))
sys.exit(0 if not failures else 1)
'''


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def probe() -> Tuple[bool, str]:
    """Run the verification in a fresh interpreter (imports are cached)."""
    script = PROBE.replace("{root}", repr(str(ROOT)).strip("'"))
    fd, path = tempfile.mkstemp(suffix=".py", prefix="trim_probe_")
    os.close(fd)
    out_fd, out_path = tempfile.mkstemp(suffix=".log", prefix="trim_probe_")
    os.close(out_fd)
    try:
        Path(path).write_text(script, encoding="utf-8")
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT)
        env["PYTHONUTF8"] = "1"
        with open(out_path, "wb") as fh:
            # stdout goes to a FILE: this sandbox forbids stdio pipes.
            subprocess.run([sys.executable, path], stdout=fh,
                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                           env=env, cwd=str(ROOT))
        text = Path(out_path).read_text(encoding="utf-8", errors="replace")
        return "PROBE_RESULT=PASS" in text, text
    finally:
        for p in (path, out_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true",
                    help="actually delete the groups that pass (default: test only)")
    ap.add_argument("--only", default="", help="comma-separated group names")
    args = ap.parse_args()

    if not SITE.is_dir():
        print(f"site-packages not found: {SITE}")
        return 1

    wanted = [g.strip() for g in args.only.split(",") if g.strip()] or list(GROUPS)
    print("=" * 70)
    print("  TRIM ENVIRONMENT" + ("" if args.apply else "   (dry run)"))
    print("=" * 70)

    print("\n  baseline probe (everything present)...")
    ok, log = probe()
    if not ok:
        print("  ! the environment is ALREADY broken; refusing to remove anything")
        print(log[-2500:])
        return 1
    print("  baseline OK\n")

    freed = 0
    kept: List[str] = []
    removed: List[str] = []

    for group in wanted:
        if group not in GROUPS:
            print(f"  unknown group: {group}")
            continue
        names = GROUPS[group]
        present = [SITE / n for n in names if (SITE / n).is_dir()]
        if not present:
            print(f"  {group:<14} already absent")
            continue

        size = sum(dir_size(p) for p in present)
        moved: List[Tuple[Path, Path]] = []
        for p in present:
            off = p.with_name(p.name + ".off")
            if off.exists():
                shutil.rmtree(off, ignore_errors=True)
            try:
                p.rename(off)
                moved.append((p, off))
            except OSError as e:
                print(f"  {group:<14} could not disable {p.name}: {e}")
                break

        ok, log = probe()
        if ok:
            freed += size
            removed.append(group)
            print(f"  {group:<14} {size / 1048576:7.1f} MB  SAFE to remove")
            if args.apply:
                for _orig, off in moved:
                    shutil.rmtree(off, ignore_errors=True)
            else:
                for orig, off in moved:   # put it back, dry run
                    off.rename(orig)
        else:
            for orig, off in moved:
                if off.exists():
                    off.rename(orig)
            kept.append(group)
            bad = [ln for ln in log.splitlines() if "FAIL" in ln]
            print(f"  {group:<14} {size / 1048576:7.1f} MB  NEEDED - restored")
            for ln in bad[:3]:
                print(f"                 {ln.strip()[:90]}")

    print()
    print(f"  reclaimable : {freed / 1048576:.1f} MB")
    if removed:
        print(f"  safe       : {', '.join(removed)}")
    if kept:
        print(f"  required   : {', '.join(kept)}")
    if not args.apply and freed:
        print("\n  re-run with --apply to actually delete")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
