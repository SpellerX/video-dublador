"""Probe which subprocess stdio modes work under the DSH file sandbox.

The sandbox blocks named pipes, so this determines whether the pipeline can
capture ffmpeg output via pipes or must redirect to temporary files.
"""
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FFMPEG = os.path.join(ROOT, ".tools", "ffmpeg", "bin", "ffmpeg.exe")

print("ffmpeg exists:", os.path.exists(FFMPEG))
results = {}


def probe(name, fn):
    try:
        fn()
        results[name] = "OK"
    except Exception as e:  # noqa: BLE001
        results[name] = f"FAIL ({type(e).__name__}: {e})"


def via_pipe_capture():
    p = subprocess.run([FFMPEG, "-version"], capture_output=True, text=True, timeout=60)
    assert "ffmpeg version" in p.stdout, "no stdout captured"
    assert p.returncode == 0


def via_pipe_stdout_only():
    p = subprocess.run([FFMPEG, "-version"], stdout=subprocess.PIPE,
                       stderr=subprocess.DEVNULL, text=True, timeout=60)
    assert "ffmpeg version" in p.stdout


def via_file_redirect():
    fd, path = tempfile.mkstemp(suffix=".txt")
    os.close(fd)
    with open(path, "w", encoding="utf-8") as fh:
        p = subprocess.run([FFMPEG, "-version"], stdout=fh, stderr=subprocess.STDOUT, timeout=60)
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        data = fh.read()
    os.unlink(path)
    assert "ffmpeg version" in data, "no output in file"
    assert p.returncode == 0


def via_devnull():
    p = subprocess.run([FFMPEG, "-version"], stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL, timeout=60)
    assert p.returncode == 0


def via_inherit():
    p = subprocess.run([FFMPEG, "-version"], timeout=60)
    assert p.returncode == 0


probe("pipe_capture_output", via_pipe_capture)
probe("pipe_stdout_only", via_pipe_stdout_only)
probe("file_redirect", via_file_redirect)
probe("devnull", via_devnull)

print()
for k, v in results.items():
    print(f"  {k:24s} -> {v}")

print()
print("RECOMMENDED_MODE=" + ("pipe" if results["pipe_capture_output"] == "OK" else "file"))
sys.exit(0)
