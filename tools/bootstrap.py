#!/usr/bin/env python
"""Provision (or repair) the portable environment for the video dubber.

Run it with the project's own interpreter::

    .python\\python.exe tools\\bootstrap.py
    .python\\python.exe tools\\bootstrap.py --packages-only
    .python\\python.exe tools\\bootstrap.py --check

It is idempotent: safe to re-run, and it only installs what is missing.

Two host quirks are handled here, both documented in README.md section 7:

* pip cannot build sdists on this machine (the sandbox denies the stdio pipes
  the PEP 517 build backend needs), so the two required sdist-only packages are
  vendored from their tarballs by ``tools/manual_install.py``.
* ``torchaudio`` >= 2.9 needs TorchCodec for audio I/O; we substitute a
  soundfile-backed shim at runtime instead (``dublador/compat.py``).
"""
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

PYTHON = ROOT / ".python" / "python.exe"
TOOLS = ROOT / ".tools"
FFMPEG_DIR = TOOLS / "ffmpeg"
FFMPEG_BIN = FFMPEG_DIR / "bin" / "ffmpeg.exe"
DIST = TOOLS / "dist"

FFMPEG_URL = "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip"

#: sdist-only, pure-Python packages that pip cannot build here.
#: (distribution name, version, module name to test for)
MANUAL_PACKAGES: List[Tuple[str, str, str]] = [
    ("antlr4-python3-runtime", "4.9.3", "antlr4"),   # pinned by omegaconf/hydra-core
    ("encodec", "0.1.1", "encodec"),                 # required by vocos
]

#: Everything installable from a wheel, in dependency-safe order.
WHEEL_PACKAGES: List[str] = [
    # audio core
    "numpy", "scipy", "soundfile", "librosa",
    # speech to text
    "faster-whisper",
    # translation
    "deep-translator",
    # F5-TTS inference dependencies (its own metadata also pulls gradio,
    # bitsandbytes, torchcodec and transformers_stream_generator, none of which
    # are needed for inference and two of which cannot be installed here)
    "cached_path", "hydra-core", "omegaconf", "matplotlib", "pydub",
    "transformers<5", "vocos", "torchdiffeq", "x_transformers",
    "rjieba", "pypinyin", "safetensors", "ema_pytorch", "accelerate",
    "datasets", "wandb", "unidecode", "tomli",
    # lip sync
    "opencv-python-headless", "tqdm",
]

TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
TORCH_PACKAGES = ["torch", "torchvision", "torchaudio"]

PIP_FLAGS = [
    "--no-cache-dir",
    "--disable-pip-version-check",
    "--no-warn-script-location",
    "--timeout", "120",
    "--retries", "5",
]


def say(msg: str = "") -> None:
    print(msg, flush=True)


def head(msg: str) -> None:
    say()
    say("=" * 68)
    say(f"  {msg}")
    say("=" * 68)


def pip(*args: str, check: bool = True) -> int:
    cmd = [str(PYTHON), "-m", "pip", "install", *PIP_FLAGS, *args]
    say(f"  $ pip install {' '.join(args)}")
    rc = subprocess.call(cmd, cwd=str(ROOT))
    if check and rc != 0:
        say(f"  ! pip exited with {rc}")
    return rc


def module_ok(name: str) -> bool:
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def have_ffmpeg() -> bool:
    return FFMPEG_BIN.exists()


def ensure_path_file() -> None:
    """Expose the project root to the embeddable interpreter.

    A ``._pth`` file makes CPython ignore ``PYTHONPATH`` entirely (only the
    ``._pth`` entries and site-packages end up on ``sys.path``), so ``import
    dublador`` would fail no matter what the launcher exports.  The supported
    way to extend the path is a ``.pth`` file inside site-packages, which
    ``site`` processes because our ``._pth`` ends with ``import site``.
    """
    sp = PYTHON.parent / "Lib" / "site-packages"
    if not sp.is_dir():
        say(f"  ! site-packages not found at {sp}")
        return
    pth = sp / "dublador_project.pth"
    want = str(ROOT)
    try:
        current = pth.read_text(encoding="utf-8", errors="replace").strip() if pth.exists() else ""
        if current != want:
            pth.write_text(want + "\n", encoding="ascii")
            say(f"  wrote {pth.name} -> {want}")
        else:
            say(f"  {pth.name} already points at the project root")
    except OSError as e:
        say(f"  ! could not write {pth}: {e}")


def install_ffmpeg() -> None:
    head("FFMPEG")
    if have_ffmpeg():
        say(f"  already present: {FFMPEG_BIN}")
        return

    DIST.mkdir(parents=True, exist_ok=True)
    archive = DIST / "ffmpeg.zip"
    if not archive.exists():
        say(f"  downloading {FFMPEG_URL}")
        urllib.request.urlretrieve(FFMPEG_URL, archive)
    say(f"  extracting ({archive.stat().st_size / 1048576:.0f} MB)")
    with zipfile.ZipFile(archive) as z:
        z.extractall(TOOLS / "_fftmp")

    inner = next(p for p in (TOOLS / "_fftmp").iterdir() if p.is_dir())
    if FFMPEG_DIR.exists():
        shutil.rmtree(FFMPEG_DIR, ignore_errors=True)
    shutil.move(str(inner), str(FFMPEG_DIR))
    shutil.rmtree(TOOLS / "_fftmp", ignore_errors=True)
    say(f"  installed: {FFMPEG_BIN}")


def install_wheels() -> None:
    head("PYTHON PACKAGES (wheels)")
    pip("--upgrade", "pip", "setuptools", "wheel", check=False)

    if not module_ok("torch"):
        say("  torch is missing - installing the CPU build (this is the big one)")
        pip(*TORCH_PACKAGES, "--index-url", TORCH_INDEX, check=False)
    else:
        import torch  # noqa: PLC0415

        say(f"  torch already present: {torch.__version__} (cuda={torch.cuda.is_available()})")
        for extra in ("torchvision", "torchaudio"):
            if not module_ok(extra):
                say(f"  installing missing {extra}")
                pip(extra, "--index-url", TORCH_INDEX, check=False)

    pip(*WHEEL_PACKAGES, check=False)

    if not module_ok("f5_tts"):
        say("  installing f5-tts without its (inference-irrelevant) metadata deps")
        pip("--no-deps", "f5-tts", check=False)


def install_manual() -> None:
    head("SDIST-ONLY PACKAGES (vendored)")
    from tools.manual_install import main as manual_main  # noqa: PLC0415

    for name, version, mod in MANUAL_PACKAGES:
        if module_ok(mod):
            say(f"  {name} {version}: already importable")
            continue
        say(f"  {name} {version}: vendoring from sdist")
        sys.argv = ["manual_install.py", name, version]
        manual_main()


def install_lipsync() -> bool:
    head("LIP SYNC (Wav2Lip)")
    say("  Wav2Lip's own weights are fetched on demand by dublador/lipsync.py.")
    ok = True
    for pkg in ("cv2", "tqdm"):
        if not module_ok(pkg):
            say(f"  installing {pkg}")
            pip("--no-deps", "opencv-python-headless" if pkg == "cv2" else "tqdm", check=False)
    if not module_ok("cv2"):
        say("  ! OpenCV is unavailable - lip sync will fall back to a plain mux")
        ok = False
    return ok


def verify() -> bool:
    head("VERIFICATION")
    try:
        from dublador.cli import environment_report  # noqa: PLC0415

        rep = environment_report()
    except Exception as e:  # noqa: BLE001
        say(f"  ! could not run the environment report: {e}")
        return False

    width = max(len(k) for k in rep["checks"])
    for name, st in rep["checks"].items():
        mark = "OK  " if st["ok"] else "MISS"
        say(f"  [{mark}] {name:<{width}}  {st['detail']}")

    bad = [k for k, v in rep["checks"].items() if not v["ok"]]
    say()
    if bad:
        say(f"  {len(bad)} component(s) unavailable: {', '.join(bad)}")
        say("  The pipeline degrades gracefully, but quality/features will suffer.")
        return False
    say("  Environment is complete.")
    return True


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ffmpeg-only", action="store_true")
    ap.add_argument("--packages-only", action="store_true")
    ap.add_argument("--lipsync-only", action="store_true")
    ap.add_argument("--check", action="store_true", help="only run the environment report")
    ap.add_argument("--no-lipsync", action="store_true", help="skip the lip-sync extras")
    args = ap.parse_args(argv)

    head("VIDEO DUBBER BOOTSTRAP")
    say(f"  project : {ROOT}")
    say(f"  python  : {sys.executable}")
    say(f"  version : {sys.version.split()[0]}")

    if not PYTHON.exists():
        say()
        say("  ! The portable interpreter .python/python.exe is missing.")
        say("    Create it first (see README section 1), then re-run this script.")
        return 1

    if args.check:
        ensure_path_file()
        return 0 if verify() else 1
    if args.ffmpeg_only:
        install_ffmpeg()
        return 0
    if args.lipsync_only:
        return 0 if install_lipsync() else 1
    if args.packages_only:
        ensure_path_file()
        install_wheels()
        install_manual()
        return 0 if verify() else 1

    ensure_path_file()
    install_ffmpeg()
    install_wheels()
    install_manual()
    if not args.no_lipsync:
        install_lipsync()
    ok = verify()

    head("DONE")
    say("  Try:  .\\run.ps1 check")
    say("        .\\run.ps1 input\\video.mp4 --target pt")
    say()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
