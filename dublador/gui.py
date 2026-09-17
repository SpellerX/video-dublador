"""Local web interface for the video dubber.

Why a web UI and not Tkinter: this project runs on the portable embeddable
Python, which ships **no tkinter** (no tcl/tk).  A local HTTP server built on
the standard library needs nothing extra, works offline, streams live progress
into the browser and gives us a real file picker for free.

Design notes
------------
* The pipeline runs in a **child process** with its stdout redirected to a
  FILE.  This sandbox forbids stdio pipes, so ``subprocess.Popen`` is never
  given ``PIPE`` -- the same rule the rest of the project follows.
* Every job is checkpointed by the pipeline itself, so cancelling or crashing
  the UI never loses completed stages.
* The server binds loopback only and requires a per-run token on every mutating
  request, so a stray web page cannot drive the local API.

    .\\abrir-interface.cmd          (double-click)
    .\\run.ps1 gui --port 8760
"""
from __future__ import annotations

import json
import mimetypes
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from .config import (INPUT_DIR, LANGUAGES, MODELS_DIR, OUTPUT_DIR, PROJECT_ROOT,
                     WORK_DIR, ensure_dirs, normalize_lang, tool_env)
from .utils import LOG, human_duration, setup_logging

STATIC_DIR = Path(__file__).resolve().parent / "static"
GUI_DIR = WORK_DIR / "gui"

VIDEO_EXT = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv", ".flv", ".ts"}
MAX_UPLOAD = 4 * 1024 ** 3  # 4 GB


# --------------------------------------------------------------------------
# job model
# --------------------------------------------------------------------------
@dataclass
class Job:
    id: str
    video: str
    target: str
    args: List[str]
    job_dir: str
    log_path: str
    created: float = field(default_factory=time.time)
    started: Optional[float] = None
    finished: Optional[float] = None
    status: str = "queued"          # queued|running|done|failed|cancelled
    returncode: Optional[int] = None
    output: Optional[str] = None
    error: Optional[str] = None
    _proc: Optional[subprocess.Popen] = field(default=None, repr=False, compare=False)

    # -- derived -----------------------------------------------------------
    @property
    def elapsed(self) -> float:
        end = self.finished or time.time()
        return max(0.0, end - (self.started or self.created))

    def to_dict(self) -> Dict[str, Any]:
        """Serialise explicitly.

        ``dataclasses.asdict`` deep-copies every field, which explodes on the
        live ``subprocess.Popen`` handle (it holds thread locks:
        "cannot pickle '_thread.lock' object").  So build the dict by hand and
        never touch ``_proc``.
        """
        return {
            "id": self.id,
            "video": self.video,
            "target": self.target,
            "args": list(self.args),
            "job_dir": self.job_dir,
            "log_path": self.log_path,
            "created": self.created,
            "started": self.started,
            "finished": self.finished,
            "status": self.status,
            "returncode": self.returncode,
            "output": self.output,
            "error": self.error,
            "elapsed": round(self.elapsed, 1),
            "elapsed_human": human_duration(self.elapsed),
        }


class JobManager:
    """Starts, tracks and cancels pipeline runs."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._jobs: Dict[str, Job] = {}
        GUI_DIR.mkdir(parents=True, exist_ok=True)
        self._load_existing()

    # -- persistence -------------------------------------------------------
    def _load_existing(self) -> None:
        """Show jobs from previous server runs (as interrupted or finished)."""
        for status_file in sorted(GUI_DIR.glob("*/status.json")):
            try:
                data = json.loads(status_file.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            job = Job(
                id=data["id"], video=data.get("video", ""), target=data.get("target", ""),
                args=data.get("args", []), job_dir=str(status_file.parent),
                log_path=data.get("log_path", str(status_file.parent / "run.log")),
            )
            job.created = data.get("created", job.created)
            job.started = data.get("started")
            job.finished = data.get("finished")
            job.status = data.get("status", "unknown")
            job.returncode = data.get("returncode")
            job.output = data.get("output")
            job.error = data.get("error")
            if job.status in ("running", "queued"):
                job.status = "interrupted"
                job.error = "the interface was closed while this job was running"
            self._jobs[job.id] = job

    def _persist(self, job: Job) -> None:
        try:
            p = Path(job.job_dir) / "status.json"
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(job.to_dict(), indent=2, ensure_ascii=False),
                           encoding="utf-8")
            tmp.replace(p)
        except OSError as e:
            LOG.debug("could not persist job %s: %s", job.id, e)

    # -- lifecycle ---------------------------------------------------------
    def start(self, config: Dict[str, Any]) -> Job:
        video = str(config.get("video") or "").strip()
        if not video:
            raise ValueError("no video selected")
        target = normalize_lang(str(config.get("target") or "pt"))

        src = Path(video)
        if not src.is_absolute():
            src = INPUT_DIR / video
        if not src.exists():
            raise FileNotFoundError(f"video not found: {src}")

        job_id = time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(3)
        job_dir = GUI_DIR / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        log_path = job_dir / "run.log"

        args = build_cli_args(src, config)

        job = Job(
            id=job_id, video=str(src), target=target, args=[str(a) for a in args],
            job_dir=str(job_dir), log_path=str(log_path),
            status="running", started=time.time(),
        )

        # stdout goes to a FILE -- never a pipe (the sandbox denies pipes).
        log_fh = open(log_path, "wb")
        try:
            env = tool_env()
            env["PYTHONPATH"] = str(PROJECT_ROOT)
            proc = subprocess.Popen(
                [sys.executable, "-m", "dublador", *[str(a) for a in args]],
                cwd=str(PROJECT_ROOT),
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=env,
            )
        except Exception as e:  # noqa: BLE001
            log_fh.close()
            job.status = "failed"
            job.error = f"could not start the pipeline: {e}"
            job.finished = time.time()
            with self._lock:
                self._jobs[job_id] = job
            self._persist(job)
            raise

        log_fh.close()  # the child owns the handle now
        job._proc = proc

        with self._lock:
            self._jobs[job_id] = job
        self._persist(job)

        threading.Thread(target=self._watch, args=(job,), daemon=True).start()
        LOG.info("job %s started: %s", job_id, " ".join(job.args[:6]))
        return job

    def _watch(self, job: Job) -> None:
        proc = job._proc
        if proc is None:
            return
        code = proc.wait()
        job.finished = time.time()
        job.returncode = code

        if job.status == "cancelled":
            job.error = "cancelled by the user"
        elif code == 0:
            job.status = "done"
            job.output = self._find_output(job)
        else:
            job.status = "failed"
            job.error = self._tail_error(job)

        self._persist(job)
        LOG.info("job %s finished: %s (exit %s)", job.id, job.status, code)

    @staticmethod
    def _find_output(job: Job) -> Optional[str]:
        """Ask the pipeline's manifest where the video went."""
        manifest = WORK_DIR / f"{Path(job.video).stem}_{job.target}" / "manifest.json"
        try:
            data = json.loads(manifest.read_text(encoding="utf-8"))
            out = data.get("output")
            if out and Path(out).exists():
                return out
        except Exception:  # noqa: BLE001
            pass
        # fall back to the newest matching file in output/
        candidates = sorted(OUTPUT_DIR.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        return str(candidates[0]) if candidates else None

    @staticmethod
    def _tail_error(job: Job) -> str:
        try:
            text = Path(job.log_path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return "the job failed"
        lines = [ln for ln in text.splitlines() if "ERROR" in ln or "FAILED" in ln]
        if lines:
            return lines[-1].split("ERROR", 1)[-1].strip(" :") or lines[-1]
        tail = text.strip().splitlines()[-1:] or ["the job failed"]
        return tail[0]

    # -- queries -----------------------------------------------------------
    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self) -> List[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda j: j.created, reverse=True)

    def cancel(self, job_id: str) -> bool:
        job = self.get(job_id)
        if not job or job.status not in ("running", "queued"):
            return False
        proc = job._proc
        job.status = "cancelled"
        if proc and proc.poll() is None:
            try:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            except Exception as e:  # noqa: BLE001
                LOG.warning("could not stop job %s: %s", job_id, e)
                return False
        self._persist(job)
        LOG.info("job %s cancelled", job_id)
        return True

    def shutdown(self) -> None:
        for job in self.list():
            if job.status in ("running", "queued"):
                self.cancel(job.id)


# --------------------------------------------------------------------------
# CLI argument construction
# --------------------------------------------------------------------------
def build_cli_args(video: Path, cfg: Dict[str, Any]) -> List[str]:
    """Translate the UI form into ``python -m dublador`` arguments."""
    args: List[str] = [str(video)]
    args += ["--target", normalize_lang(str(cfg.get("target") or "pt"))]

    source = str(cfg.get("source") or "").strip()
    if source and source != "auto":
        args += ["--source", normalize_lang(source)]

    # The UI's preset buttons write their values straight into the form fields,
    # so we send the *effective* values and never the preset flag: passing both
    # would silently contradict itself (argparse is last-wins) the moment a user
    # tweaks one field after picking a preset.

    # Explicit values (the UI always sends what it currently shows).
    simple = {
        "whisper_model": "--whisper-model",
        "whisper_compute_type": "--whisper-compute-type",
        "translator": "--translator",
        "f5_model": "--f5-model",
        "background": "--background",
        "device": "--device",
        "until": "--until",
    }
    for key, flag in simple.items():
        val = cfg.get(key)
        if val not in (None, "", "auto") or key == "translator":
            if val not in (None, "") and not (key == "until" and val == "all"):
                args += [flag, str(val)]

    numeric = {
        "nfe_step": "--nfe-step",
        "beam_size": "--beam-size",
        "min_speakers": "--min-speakers",
        "max_speakers": "--max-speakers",
        "whisper_model_threads": "--threads",
        "lipsync_resize": "--lipsync-resize",
    }
    for key, flag in numeric.items():
        val = cfg.get(key)
        if val not in (None, ""):
            try:
                args += [flag, str(int(val))]
            except (TypeError, ValueError):
                pass

    if cfg.get("no_lipsync") is True:
        args += ["--no-lipsync"]
    if cfg.get("lipsync_force") is True:
        args += ["--lipsync-force"]
    if cfg.get("no_vad") is True:
        args += ["--no-vad"]
    if cfg.get("no_fix_duration") is True:
        args += ["--no-fix-duration"]
    if cfg.get("verbose") is True:
        args += ["-v"]

    # ---- naturalisation ------------------------------------------------
    if cfg.get("naturalize") is False:
        args += ["--no-naturalize"]
    if cfg.get("no_fit_slots") is True:
        args += ["--no-fit-slots"]

    text_opts = {
        "register": "--register",
        "glossary": "--glossary",
        "llm_url": "--llm-url",
        "llm_model": "--llm-model",
        "llm_key": "--llm-key",
    }
    for key, flag in text_opts.items():
        val = cfg.get(key)
        if isinstance(val, str) and val.strip():
            args += [flag, val.strip()]
    if cfg.get("translate_context"):
        try:
            ctx = int(cfg["translate_context"])
            if ctx > 1:
                args += ["--translate-context", str(ctx)]
        except (TypeError, ValueError):
            pass
    if cfg.get("llm_ollama") is True:
        args += ["--llm-ollama"]
    return args


# --------------------------------------------------------------------------
# log parsing / progress
# --------------------------------------------------------------------------
_STAGE_RE = re.compile(r"STAGE\s+(\d)/7\s+(.*)")
_PCT_RE = re.compile(r"(\d{1,3}(?:\.\d)?)%")
_TOTAL_STAGES = 7


def analyze_log(text: str) -> Dict[str, Any]:
    """Best-effort progress for the UI, derived from the pipeline's own log."""
    stage_no = 0
    stage_label = "preparando"
    inner_pct = 0.0
    last_line = ""

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        last_line = line
        m = _STAGE_RE.search(line)
        if m:
            stage_no = int(m.group(1))
            stage_label = m.group(2).strip().title()
            inner_pct = 0.0
            continue
        p = _PCT_RE.search(line)
        if p and ("%") in line:
            try:
                inner_pct = max(inner_pct, float(p.group(1)))
            except ValueError:
                pass

    if "DONE" in text and "dubbed video" in text:
        overall = 100.0
    else:
        overall = ((stage_no - 1) + inner_pct / 100.0) / _TOTAL_STAGES * 100.0
    overall = max(0.0, min(99.0 if stage_no else 0.0, overall))

    return {
        "stage_number": stage_no,
        "stage_total": _TOTAL_STAGES,
        "stage_label": stage_label,
        "percent": round(overall, 1),
        "last_line": last_line[:200],
    }


# --------------------------------------------------------------------------
# HTTP handler
# --------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "dublador-gui/1.0"
    manager: JobManager
    token: str

    # -- plumbing ----------------------------------------------------------
    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003
        LOG.debug("gui %s - %s", self.address_string(), fmt % args)

    def _send(self, code: int, body: bytes, ctype: str = "application/json; charset=utf-8",
              extra: Optional[Dict[str, str]] = None) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionAbortedError):
            pass

    def _json(self, obj: Any, code: int = 200) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"))

    def _error(self, code: int, message: str) -> None:
        self._json({"ok": False, "error": message}, code)

    def _ok_token(self) -> bool:
        return self.headers.get("X-Token", "") == self.token

    # -- routes ------------------------------------------------------------
    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        try:
            if path in ("/", "/index.html"):
                return self._serve_index()
            if path == "/api/env":
                return self._api_env()
            if path == "/api/languages":
                return self._api_languages()
            if path == "/api/videos":
                return self._api_videos()
            if path == "/api/hardware":
                return self._api_hardware()
            if path == "/api/jobs":
                return self._json({"ok": True, "jobs": [j.to_dict() for j in self.manager.list()]})
            if path.startswith("/api/jobs/"):
                return self._api_job(path)
            if path.startswith("/api/doctor"):
                return self._api_doctor()
            if path.startswith("/api/file"):
                return self._api_file(parse_qs(urlparse(self.path).query))
            return self._error(404, "not found")
        except Exception as e:  # noqa: BLE001
            LOG.exception("GET %s failed", path)
            return self._error(500, str(e))

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if not self._ok_token():
            return self._error(403, "invalid or missing token")
        try:
            if path == "/api/upload":
                return self._api_upload()
            if path == "/api/jobs":
                return self._api_start_job()
            if path.startswith("/api/jobs/") and path.endswith("/cancel"):
                return self._api_cancel(path)
            return self._error(404, "not found")
        except Exception as e:  # noqa: BLE001
            LOG.exception("POST %s failed", path)
            return self._error(500, str(e))

    # -- handlers ----------------------------------------------------------
    def _serve_index(self) -> None:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            return self._error(500, f"missing {index}")
        html = index.read_text(encoding="utf-8").replace("__TOKEN__", self.token)
        return self._send(200, html.encode("utf-8"), "text/html; charset=utf-8")

    def _api_env(self) -> None:
        from .cli import environment_report  # noqa: PLC0415

        rep = environment_report()
        return self._json({"ok": True, "python": rep["python"], "checks": rep["checks"]})

    def _api_doctor(self) -> None:
        from .cli import environment_report  # noqa: PLC0415

        return self._json({"ok": True, **environment_report()})

    def _api_hardware(self) -> None:
        from .hardware import detect, recommend, report_lines  # noqa: PLC0415

        hw = detect(deep=True)
        rec = recommend(hw)
        return self._json({
            "ok": True,
            "hardware": hw.to_dict(),
            "recommendation": rec.to_dict(),
            "report": report_lines(hw, rec),
        })

    def _api_languages(self) -> None:
        return self._json({
            "ok": True,
            "languages": [{"code": c, "name": n} for c, n in
                          sorted(LANGUAGES.items(), key=lambda kv: kv[1])],
        })

    def _api_videos(self) -> None:
        ensure_dirs()
        vids = []
        for p in sorted(INPUT_DIR.iterdir(), key=lambda q: q.stat().st_mtime, reverse=True):
            if p.is_file() and p.suffix.lower() in VIDEO_EXT:
                vids.append({
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / 1048576, 2),
                    "mtime": p.stat().st_mtime,
                })
        outs = []
        for p in sorted(OUTPUT_DIR.glob("*"), key=lambda q: q.stat().st_mtime, reverse=True):
            if p.is_file():
                outs.append({
                    "name": p.name,
                    "size_mb": round(p.stat().st_size / 1048576, 2),
                    "mtime": p.stat().st_mtime,
                })
        return self._json({"ok": True, "videos": vids, "outputs": outs,
                           "input_dir": str(INPUT_DIR), "output_dir": str(OUTPUT_DIR)})

    def _api_job(self, path: str) -> None:
        parts = [p for p in path.split("/") if p]
        # /api/jobs/<id>[/log]
        if len(parts) < 3:
            return self._error(400, "missing job id")
        job = self.manager.get(parts[2])
        if not job:
            return self._error(404, "unknown job")

        if len(parts) >= 4 and parts[3] == "log":
            qs = parse_qs(urlparse(self.path).query)
            try:
                offset = int(qs.get("offset", ["0"])[0])
            except ValueError:
                offset = 0
            try:
                data = Path(job.log_path).read_bytes()
            except OSError:
                data = b""
            chunk = data[offset:]
            payload = {
                "ok": True,
                "text": chunk.decode("utf-8", errors="replace"),
                "offset": len(data),
                "eof": True,
            }
            return self._json(payload)

        detail = job.to_dict()
        try:
            text = Path(job.log_path).read_text(encoding="utf-8", errors="replace")[-40000:]
        except OSError:
            text = ""
        detail["progress"] = analyze_log(text) if text else {
            "stage_number": 0, "stage_total": _TOTAL_STAGES,
            "stage_label": "aguardando", "percent": 0.0, "last_line": "",
        }
        return self._json({"ok": True, "job": detail})

    def _api_file(self, qs: Dict[str, List[str]]) -> None:
        """Download an output video (or a subtitle file) by name."""
        name = (qs.get("name") or [""])[0]
        kind = (qs.get("kind") or ["output"])[0]
        if not name or "/" in name or "\\" in name or ".." in name:
            return self._error(400, "invalid name")

        base = OUTPUT_DIR if kind == "output" else (WORK_DIR / name.split("::")[0] if "::" in name else WORK_DIR)
        target = (base / name.split("::")[-1]).resolve()
        if not str(target).startswith(str(PROJECT_ROOT.resolve())):
            return self._error(403, "path outside the project")
        if not target.is_file():
            return self._error(404, "file not found")

        ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        data = target.read_bytes()
        return self._send(200, data, ctype, {
            "Content-Disposition": f'attachment; filename="{target.name}"',
        })

    def _api_upload(self) -> None:
        name = self.headers.get("X-Filename", "").strip()
        if not name:
            return self._error(400, "missing X-Filename header")
        name = Path(name).name  # strip any path component
        if Path(name).suffix.lower() not in VIDEO_EXT:
            return self._error(400, f"unsupported file type: {Path(name).suffix}")

        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return self._error(400, "invalid Content-Length")
        if length <= 0:
            return self._error(400, "empty upload")
        if length > MAX_UPLOAD:
            return self._error(413, "file too large (limit 4 GB)")

        ensure_dirs()
        dest = INPUT_DIR / name
        if dest.exists():
            stem, suffix = dest.stem, dest.suffix
            for i in range(1, 10000):
                cand = INPUT_DIR / f"{stem}_{i}{suffix}"
                if not cand.exists():
                    dest = cand
                    break

        remaining = length
        with open(dest, "wb") as fh:
            while remaining > 0:
                chunk = self.rfile.read(min(1 << 20, remaining))
                if not chunk:
                    break
                fh.write(chunk)
                remaining -= len(chunk)

        LOG.info("uploaded %s (%.1f MB)", dest.name, dest.stat().st_size / 1048576)
        return self._json({"ok": True, "name": dest.name,
                           "size_mb": round(dest.stat().st_size / 1048576, 2)})

    def _api_start_job(self) -> None:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        raw = self.rfile.read(length) if length else b"{}"
        try:
            cfg = json.loads(raw.decode("utf-8") or "{}")
        except json.JSONDecodeError as e:
            return self._error(400, f"invalid JSON: {e}")

        try:
            job = self.manager.start(cfg)
        except (ValueError, FileNotFoundError) as e:
            return self._error(400, str(e))
        except Exception as e:  # noqa: BLE001
            return self._error(500, str(e))
        return self._json({"ok": True, "job": job.to_dict()})

    def _api_cancel(self, path: str) -> None:
        parts = [p for p in path.split("/") if p]
        if len(parts) < 3:
            return self._error(400, "missing job id")
        ok = self.manager.cancel(parts[2])
        return self._json({"ok": ok})


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def _find_port(host: str, preferred: int) -> int:
    import socket  # noqa: PLC0415

    for port in [preferred, *range(preferred + 1, preferred + 25)]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind((host, port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"no free port near {preferred}")


def serve(host: str = "127.0.0.1", port: int = 8760, *, open_browser: bool = True,
          verbose: bool = False) -> int:
    ensure_dirs()
    setup_logging(verbose, PROJECT_ROOT / "logs" / "gui.log")

    if not (STATIC_DIR / "index.html").exists():
        LOG.error("interface assets are missing: %s", STATIC_DIR / "index.html")
        return 1

    port = _find_port(host, port)
    token = secrets.token_urlsafe(24)
    manager = JobManager()

    handler = type("BoundHandler", (Handler,), {"manager": manager, "token": token})
    httpd = ThreadingHTTPServer((host, port), handler)
    httpd.daemon_threads = True

    url = f"http://{host}:{port}/"
    LOG.info("")
    LOG.info("=" * 68)
    LOG.info("  VIDEO DUBBER - interface")
    LOG.info("=" * 68)
    LOG.info("  abra no navegador: %s", url)
    LOG.info("")
    LOG.info("  Esta janela precisa ficar aberta enquanto voce usa a interface.")
    LOG.info("  Feche com Ctrl+C.")
    LOG.info("")

    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        LOG.info("")
        LOG.info("encerrando...")
    finally:
        manager.shutdown()
        httpd.shutdown()
        httpd.server_close()
    return 0
