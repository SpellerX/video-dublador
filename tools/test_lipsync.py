"""Throwaway smoke test for ``dublador.lipsync`` (Wav2Lip).

Runs three checks and prints real, observed results:

1. weights download / verification;
2. a synthetic ``testsrc`` clip with **no face** -> Wav2Lip must refuse
   cleanly and the pipeline must degrade to ``lipsync_fallback``;
3. a clip containing a **real face** (built from a public domain test image,
   panned so the face moves) -> full Wav2Lip inference must run end to end.

Usage (from the project root):
    .python\python.exe tools\test_lipsync.py
"""
from __future__ import annotations

import sys
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dublador.config import FFMPEG_BIN, WORK_DIR, ensure_dirs, tool_env  # noqa: E402
from dublador.lipsync import (  # noqa: E402
    ensure_models,
    lipsync_available,
    lipsync_fallback,
    lipsync_model_dir,
    lipsync_video,
)
from dublador.media import media_info  # noqa: E402
from dublador.utils import (  # noqa: E402
    LOG,
    banner,
    human_bytes,
    human_duration,
    run_command,
    setup_logging,
)

OUT = WORK_DIR / "lipsync_test"
FACE_IMAGE_URLS = (
    "https://raw.githubusercontent.com/opencv/opencv/4.x/samples/data/lena.jpg",
    "https://raw.githubusercontent.com/opencv/opencv/master/samples/data/lena.jpg",
)

FAILURES: list[str] = []


def ffmpeg(args: list[str], *, desc: str) -> None:
    cmd = [FFMPEG_BIN, "-hide_banner", "-nostdin", "-y", *args]
    res = run_command(cmd, env=tool_env(), desc=desc, timeout=300)
    if not res.ok:
        raise RuntimeError(f"ffmpeg {desc} failed (exit {res.returncode}):\n{res.output[-2000:]}")


def check(ok: bool, label: str, detail: str = "") -> None:
    print(f"   [{'PASS' if ok else 'FAIL'}] {label}{(' - ' + detail) if detail else ''}")
    if not ok:
        FAILURES.append(label)


def environment() -> None:
    banner("0. environment")
    import cv2
    import librosa
    import torch

    print(f"   python      : {sys.version.split()[0]} ({sys.executable})")
    print(f"   torch       : {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    print(f"   cv2         : {cv2.__version__}")
    print(f"   librosa     : {librosa.__version__}")
    print(f"   device      : cpu only, {torch.get_num_threads()} torch thread(s)")
    print(f"   lipsync_available(): {lipsync_available()}")


def step1_models() -> Path:
    banner("1. ensure_models() - download + verify weights")
    t0 = time.time()
    models = ensure_models()
    print(f"   model dir   : {models}")
    for name in ("wav2lip_gan.pth", "wav2lip.pth", "s3fd.pth"):
        p = models / name
        if p.is_file():
            print(f"   {name:18s}: {human_bytes(p.stat().st_size)}")
    check(models.is_dir(), "ensure_models returns a directory", str(models))
    check((models / "wav2lip_gan.pth").is_file(), "wav2lip_gan.pth present")
    check((models / "s3fd.pth").is_file(), "s3fd.pth present")
    print(f"   (took {human_duration(time.time() - t0)})")
    return models


def make_testsrc(path: Path) -> None:
    """320x240 testsrc + silent track: contains NO face."""
    ffmpeg(
        [
            "-f", "lavfi", "-i", "testsrc=size=320x240:rate=25",
            "-f", "lavfi", "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
            "-t", "3", "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-shortest", str(path),
        ],
        desc="testsrc",
    )


def make_dub_audio(path: Path, seconds: float = 3.0) -> None:
    """A stand-in for the synthesised dubbed voice track."""
    ffmpeg(
        ["-f", "lavfi", "-i", f"sine=frequency=220:sample_rate=16000:duration={seconds}",
         "-ac", "1", "-c:a", "pcm_s16le", str(path)],
        desc="dub-audio",
    )


def fetch_face_image(path: Path) -> bool:
    if path.is_file() and path.stat().st_size > 10_000:
        return True
    for url in FACE_IMAGE_URLS:
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "dublador-lipsync-test/1.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) > 10_000:
                path.write_bytes(data)
                print(f"   fetched face image: {url} ({human_bytes(len(data))})")
                return True
        except Exception as e:  # noqa: BLE001
            print(f"   face image mirror failed: {url} ({e})")
    return False


def make_face_video(image: Path, path: Path, seconds: float = 3.0) -> None:
    """A 25 fps clip of a real face that pans slowly across the frame."""
    vf = (
        "scale=640:640,"
        "crop=512:512:x='(iw-512)/2+40*sin(2*PI*t/3)':y='(ih-512)/2',"
        "format=yuv420p"
    )
    ffmpeg(
        ["-loop", "1", "-i", str(image), "-f", "lavfi",
         "-i", "anullsrc=channel_layout=mono:sample_rate=16000",
         "-t", str(seconds), "-r", "25", "-vf", vf,
         "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-shortest", str(path)],
        desc="face-video",
    )


def video_mad(a: Path, b: Path, frame: int = 20) -> float:
    """Mean absolute pixel difference between frame N of two videos."""
    import cv2
    import numpy as np

    out = []
    for path in (a, b):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        cap.release()
        out.append(img.astype("float32") if ok and img is not None else None)
    if out[0] is None or out[1] is None or out[0].shape != out[1].shape:
        return -1.0
    return float(np.abs(out[0] - out[1]).mean())


def report_output(label: str, path: Path) -> None:
    info = media_info(path)
    print(
        f"   {label}: {path.name}  {human_bytes(path.stat().st_size)}  "
        f"{info.resolution} {info.fps:.2f}fps {info.duration:.2f}s "
        f"v={info.video_codec} a={info.audio_codec}"
    )


def step2_downscale(dub: Path) -> None:
    banner("2. internal face-detection downscale for large frames (unit check)")
    import numpy as np

    from dublador import lipsync as L

    class Stub:
        """Returns a fixed box in whatever image it is handed."""

        def __init__(self) -> None:
            self.seen: tuple[int, ...] | None = None

        def detect_batch(self, images):  # noqa: ANN001, ANN201
            self.seen = tuple(images.shape)
            return [(100, 100, 400, 300)] * images.shape[0]

    stub = Stub()
    big = [np.zeros((1080, 1920, 3), dtype=np.uint8) for _ in range(2)]
    boxes = L._detect_chunk(stub, big, (0, 10, 0, 0))
    print(f"   1080p frames -> detector saw {stub.seen}, box {boxes[0]}")
    check(stub.seen == (2, 540, 960, 3), "1080p detection ran on a 960x540 copy", str(stub.seen))
    check(
        boxes[0] == (200, 200, 800, 610),
        "boxes scaled back to full res (x2) + bottom pad",
        str(boxes[0]),
    )

    stub2 = Stub()
    small = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(2)]
    boxes2 = L._detect_chunk(stub2, small, (0, 10, 0, 0))
    print(f"   480p frames  -> detector saw {stub2.seen}, box {boxes2[0]}")
    check(stub2.seen == (2, 480, 640, 3), "frames under the cap are not downscaled")
    check(boxes2[0] == (100, 100, 400, 310), "boxes left untouched below the cap", str(boxes2[0]))


def step3_no_face(dub: Path) -> None:
    banner("3. lipsync_video() on a synthetic testsrc clip (NO face)")
    video = OUT / "testsrc.mp4"
    make_testsrc(video)
    report_output("input ", video)

    t0 = time.time()
    raised: str | None = None
    try:
        lipsync_video(video, dub, OUT / "testsrc_lipsync.mp4")
    except RuntimeError as e:
        raised = str(e)
    except Exception as e:  # noqa: BLE001
        raised = f"UNEXPECTED {type(e).__name__}: {e}"
        traceback.print_exc()
    elapsed = time.time() - t0

    print(f"   lipsync_video() -> {raised!r}  ({human_duration(elapsed)})")
    check(raised is not None, "Wav2Lip refuses a faceless video")
    check(
        raised is not None and "no face detected" in raised.lower(),
        "refusal explains that no face was detected",
    )
    check(
        not (OUT / "testsrc_lipsync.mp4").exists(),
        "no half-written output left behind",
    )

    # The pipeline's degradation path must still deliver a usable video.
    fallback = OUT / "testsrc_fallback.mp4"
    lipsync_fallback(video, dub, fallback, reason=raised or "")
    report_output("fallback", fallback)
    info = media_info(fallback)
    check(fallback.is_file() and fallback.stat().st_size > 0, "lipsync_fallback wrote a file")
    check(info.has_video and info.has_audio, "fallback output keeps video + audio")
    check(abs(info.duration - 3.0) < 0.5, "fallback duration preserved", f"{info.duration:.2f}s")


def step4_face(dub: Path) -> None:
    banner("4. lipsync_video() on a clip WITH a real face")
    image = OUT / "lena.jpg"
    if not fetch_face_image(image):
        check(False, "could not obtain a face image for testing")
        return
    video = OUT / "face.mp4"
    make_face_video(image, video)
    report_output("input ", video)

    out = OUT / "face_lipsync.mp4"
    t0 = time.time()
    progress: list[tuple[int, int]] = []

    def cb(done: int, total: int) -> None:
        progress.append((done, total))

    try:
        result = lipsync_video(video, dub, out, batch_size=8, progress_cb=cb)
    except RuntimeError:
        traceback.print_exc()
        check(False, "Wav2Lip ran on a video that does contain a face")
        return
    elapsed = time.time() - t0

    report_output("output ", result)
    info = media_info(result)
    check(result.is_file() and result.stat().st_size > 0, "wav2lip output written")
    check(info.has_video and info.has_audio, "wav2lip output keeps video + audio")
    check(len(progress) > 0, "progress_cb was called", f"{len(progress)} call(s)")

    mad = video_mad(video, result, frame=20)
    print(f"   mean abs pixel diff vs input (frame 20): {mad:.2f}")
    check(mad > 1.0, "mouth region pixels were re-animated by the model", f"MAD={mad:.2f}")
    print(
        f"   wall clock  : {human_duration(elapsed)} for {info.duration:.1f}s of video "
        f"({info.duration / max(elapsed, 1e-6):.3f}x realtime)"
    )


def step5_static(dub: Path) -> None:
    banner("5. lipsync_video(static=True) - single-frame detection path")
    video = OUT / "face.mp4"
    out = OUT / "face_static.mp4"
    t0 = time.time()
    try:
        result = lipsync_video(video, dub, out, static=True, batch_size=8)
    except RuntimeError:
        traceback.print_exc()
        check(False, "static=True ran end to end")
        return
    report_output("static ", result)
    info = media_info(result)
    check(info.has_video and info.has_audio, "static output keeps video + audio")
    mad = video_mad(video, result, frame=20)
    print(f"   mean abs pixel diff vs input (frame 20): {mad:.2f}")
    check(mad > 1.0, "static mode re-animated pixels too", f"MAD={mad:.2f}")
    print(f"   wall clock  : {human_duration(time.time() - t0)}")


def main() -> int:
    setup_logging(verbose=False)
    ensure_dirs()
    OUT.mkdir(parents=True, exist_ok=True)
    environment()
    models = step1_models()
    print(f"   model dir stat: {models}")

    dub = OUT / "dub.wav"
    make_dub_audio(dub)

    step2_downscale(dub)
    step3_no_face(dub)
    step4_face(dub)
    step5_static(dub)

    banner("SUMMARY")
    print(f"   artefacts   : {OUT}")
    if FAILURES:
        print(f"   FAILED CHECKS ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"     - {f}")
        return 1
    print("   all checks passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
