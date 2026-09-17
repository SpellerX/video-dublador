"""Generic helpers: logging, sandbox-safe subprocess execution, JSON, cache.

IMPORTANT (host constraint)
---------------------------
The DSH file sandbox on this machine denies subprocess stdio **pipes**
(``PermissionError: [WinError 5]``), so every helper here redirects child
output into a temporary *file* and reads it back.  Never use
``subprocess.PIPE`` anywhere in this project.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

LOG = logging.getLogger("dublador")

Cmd = Sequence[Union[str, Path]]


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------
def setup_logging(verbose: bool = False, log_file: Optional[Path] = None) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)

    fmt = logging.Formatter("%(asctime)s  %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setLevel(level)
    sh.setFormatter(fmt)
    root.addHandler(sh)

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        root.addHandler(fh)

    # Third-party noise reduction.
    for noisy in ("urllib3", "filelock", "numba", "matplotlib", "huggingface_hub", "speechbrain"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return LOG


def banner(msg: str) -> None:
    LOG.info("")
    LOG.info("=" * 68)
    LOG.info("  %s", msg)
    LOG.info("=" * 68)


# --------------------------------------------------------------------------
# Subprocess (file-redirect based)
# --------------------------------------------------------------------------
class CommandError(RuntimeError):
    def __init__(self, cmd: Cmd, returncode: int, output: str):
        self.cmd = [str(c) for c in cmd]
        self.returncode = returncode
        self.output = output
        tail = "\n".join(output.strip().splitlines()[-30:])
        super().__init__(
            f"command failed (exit {returncode}):\n  {' '.join(self.cmd)}\n{tail}"
        )


@dataclass
class CommandResult:
    returncode: int
    output: str
    seconds: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run_command(
    cmd: Cmd,
    *,
    cwd: Optional[Union[str, Path]] = None,
    timeout: Optional[float] = None,
    check: bool = True,
    env: Optional[Dict[str, str]] = None,
    desc: Optional[str] = None,
) -> CommandResult:
    """Run ``cmd`` capturing combined stdout+stderr through a temp file.

    Mirrors ``subprocess.run(..., capture_output=True, text=True)`` but works
    under a sandbox that forbids pipes.
    """
    cmd = [str(c) for c in cmd]
    if desc:
        LOG.debug("run%s: %s", f" [{desc}]" if desc else "", " ".join(cmd[:8]))

    fd, out_path = tempfile.mkstemp(prefix="dublador-cmd-", suffix=".log")
    os.close(fd)
    started = time.time()
    try:
        with open(out_path, "wb") as fh:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd) if cwd else None,
                stdout=fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                timeout=timeout,
                env=env,
            )
        try:
            raw = Path(out_path).read_bytes()
        except OSError:
            raw = b""
        output = raw.decode("utf-8", errors="replace")
    except FileNotFoundError as e:
        raise CommandError(cmd, 127, f"executable not found: {cmd[0]} ({e})") from e
    except subprocess.TimeoutExpired as e:
        partial = ""
        try:
            partial = Path(out_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            pass
        raise CommandError(cmd, 124, f"timed out after {timeout}s\n{partial}") from e
    finally:
        try:
            os.unlink(out_path)
        except OSError:
            pass

    result = CommandResult(proc.returncode, output, time.time() - started)
    if check and proc.returncode != 0:
        raise CommandError(cmd, proc.returncode, output)
    return result


def run_python(args: Sequence[str], **kw) -> CommandResult:
    """Run a snippet/module with the current interpreter."""
    return run_command([sys.executable, *args], **kw)


# --------------------------------------------------------------------------
# Files / JSON
# --------------------------------------------------------------------------
def ensure_dir(p: Union[str, Path]) -> Path:
    d = Path(p)
    d.mkdir(parents=True, exist_ok=True)
    return d


def read_json(path: Union[str, Path], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        LOG.warning("could not read %s: %s", p, e)
        return default


def write_json(path: Union[str, Path], data: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    tmp.replace(p)
    return p


def file_hash(path: Union[str, Path], chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def unique_path(path: Union[str, Path]) -> Path:
    p = Path(path)
    if not p.exists():
        return p
    stem, suffix, parent = p.stem, p.suffix, p.parent
    for i in range(1, 10000):
        cand = parent / f"{stem}_{i}{suffix}"
        if not cand.exists():
            return cand
    raise RuntimeError(f"could not find a free name for {p}")


def require_file(path: Union[str, Path], what: str = "file") -> Path:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"{what} not found: {p}")
    return p


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{seconds:.1f}s"


def human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


# --------------------------------------------------------------------------
# Per-stage checkpointing (critical on slow hardware)
# --------------------------------------------------------------------------
class StageCache:
    """Skip expensive stages when their inputs have not changed.

    Usage::

        cache = StageCache(job_dir)
        if cache.hit("transcribe", inputs=[audio], outputs=[json_out]):
            data = read_json(json_out)
        else:
            data = expensive()
            write_json(json_out, data)
            cache.mark("transcribe", inputs=[audio], outputs=[json_out])
    """

    def __init__(self, job_dir: Union[str, Path], enabled: bool = True):
        self.path = Path(job_dir) / "stages.json"
        self.enabled = enabled
        self.state = read_json(self.path, default={}) or {}

    @staticmethod
    def _fingerprint(inputs: Iterable[Union[str, Path]]) -> str:
        parts = []
        for i in inputs:
            p = Path(i)
            if p.exists() and p.is_file():
                st = p.stat()
                parts.append(f"{p.name}:{st.st_size}:{int(st.st_mtime)}")
            else:
                parts.append(f"{p}:missing")
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]

    def hit(self, stage: str, inputs: Iterable[Union[str, Path]],
            outputs: Iterable[Union[str, Path]]) -> bool:
        if not self.enabled:
            return False
        rec = self.state.get(stage)
        if not rec:
            return False
        if rec.get("fingerprint") != self._fingerprint(inputs):
            return False
        for o in outputs:
            p = Path(o)
            if not p.exists() or p.stat().st_size == 0:
                return False
        LOG.info("  [cache] reusing '%s' results", stage)
        return True

    def mark(self, stage: str, inputs: Iterable[Union[str, Path]],
             outputs: Iterable[Union[str, Path]], **meta: Any) -> None:
        self.state[stage] = {
            "fingerprint": self._fingerprint(inputs),
            "outputs": [str(o) for o in outputs],
            "at": time.strftime("%Y-%m-%d %H:%M:%S"),
            **meta,
        }
        write_json(self.path, self.state)

    def clear(self) -> None:
        self.state = {}
        write_json(self.path, self.state)


class Progress:
    """Minimal ETA-aware progress logger."""

    def __init__(self, total: int, label: str = "", every: int = 1):
        self.total = max(0, total)
        self.label = label
        self.every = max(1, every)
        self.start = time.time()
        self.n = 0

    def step(self, extra: str = "") -> None:
        self.n += 1
        if self.n % self.every and self.n != self.total:
            return
        elapsed = time.time() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0
        eta = (self.total - self.n) / rate if rate > 0 else 0
        LOG.info(
            "  %s %d/%d (%.1f%%)  elapsed %s  eta %s %s",
            self.label, self.n, self.total,
            (100.0 * self.n / self.total) if self.total else 100.0,
            human_duration(elapsed), human_duration(eta), extra,
        )


def free_disk_gb(path: Union[str, Path]) -> float:
    try:
        return shutil.disk_usage(str(path)).free / (1024 ** 3)
    except OSError:
        return float("nan")
