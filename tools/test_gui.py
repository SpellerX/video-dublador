#!/usr/bin/env python
"""Exercise the local web interface end to end.

Acts like the browser: fetches the page, scrapes the per-run token out of the
HTML, then drives the API -- including starting a real pipeline job and
streaming its log.

    .python\\python.exe tools\\test_gui.py [--port 8760]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

PASS, FAIL = "PASS", "FAIL"
results = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def get(base: str, path: str, token: str | None = None):
    req = urllib.request.Request(base + path, headers={"X-Token": token or ""})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.status, r.read()


def post(base: str, path: str, payload, token: str):
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        base + path, data=body, method="POST",
        headers={"Content-Type": "application/json", "X-Token": token},
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8760)
    ap.add_argument("--video", default="test_speech.mp4")
    ap.add_argument("--skip-job", action="store_true")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    print("=" * 68)
    print(f"  GUI TEST  {base}")
    print("=" * 68)

    # ---- page + token ----------------------------------------------------
    try:
        status, body = get(base, "/")
    except Exception as e:  # noqa: BLE001
        print(f"  cannot reach the interface: {e}")
        print("  start it with:  .\\run.ps1 gui")
        return 1

    html = body.decode("utf-8", "replace")
    check("GET / returns 200", status == 200, f"{len(html)} bytes")
    check("page is the dublador UI", "Video <span>Dublador</span>" in html)
    m = re.search(r'const TOKEN = "([^"]+)"', html)
    check("token injected into page", bool(m))
    if not m:
        return 1
    token = m.group(1)
    check("token placeholder was replaced", "__TOKEN__" not in html)

    # ---- read-only API ---------------------------------------------------
    for path in ("/api/env", "/api/languages", "/api/videos", "/api/jobs", "/api/hardware"):
        try:
            st, body = get(base, path)
            data = json.loads(body)
            check(f"GET {path}", st == 200 and data.get("ok") is True)
        except Exception as e:  # noqa: BLE001
            check(f"GET {path}", False, str(e))

    st, body = get(base, "/api/hardware")
    hwdata = json.loads(body)
    hw, rec = hwdata.get("hardware", {}), hwdata.get("recommendation", {})
    check("hardware probe reports a CPU", bool(hw.get("cpu", {}).get("model")))
    check("hardware probe reports physical cores",
          int(hw.get("cpu", {}).get("physical_cores") or 0) >= 1,
          f"{hw.get('cpu', {}).get('physical_cores')}C/"
          f"{hw.get('cpu', {}).get('logical_cores')}T")
    check("hardware probe reports memory", float(hw.get("memory", {}).get("total_gb") or 0) > 0,
          f"{hw.get('memory', {}).get('total_gb')} GB")
    check("a capability tier was chosen", bool(rec.get("tier")), rec.get("tier", ""))
    check("a speed estimate was produced", float(rec.get("realtime_factor") or 0) > 0,
          f"~{rec.get('realtime_factor')}x realtime")
    check("threads capped at physical cores",
          int(rec.get("threads") or 0) <= int(hw.get("cpu", {}).get("physical_cores") or 99),
          f"{rec.get('threads')} threads")
    check("recommendation is self-consistent",
          bool(rec.get("whisper_model")) and int(rec.get("nfe_step") or 0) > 0,
          f"{rec.get('whisper_model')} @ {rec.get('nfe_step')} steps")

    st, body = get(base, "/api/videos")
    vids = json.loads(body).get("videos", [])
    check("video list is populated", any(v["name"] == args.video for v in vids),
          f"{len(vids)} video(s)")

    # ---- token enforcement ----------------------------------------------
    st, _ = post(base, "/api/jobs", {"video": args.video, "target": "pt"}, token="wrong")
    check("POST with a bad token is rejected", st == 403, f"HTTP {st}")

    if args.skip_job:
        return _summary()

    # ---- start a real job, letting the program auto-tune itself ----------
    cfg = {
        "video": args.video, "target": "pt", "source": "auto",
        "preset": "auto",
        # "auto" everywhere exercises the hardware-driven path end to end.
        "whisper_model": "auto", "nfe_step": "auto", "f5_model": "auto",
        "translator": "google", "background": "none", "until": "transcribe",
        "min_speakers": 1, "max_speakers": 6, "no_lipsync": True,
    }
    st, body = post(base, "/api/jobs", cfg, token)
    data = json.loads(body)
    check("POST /api/jobs starts a job", st == 200 and data.get("ok") is True,
          data.get("error", ""))
    if not data.get("ok"):
        return _summary()

    job_id = data["job"]["id"]
    print(f"  job id: {job_id}")

    # ---- stream the log --------------------------------------------------
    offset = 0
    seen = ""
    deadline = time.time() + 300
    final = None
    while time.time() < deadline:
        st, body = get(base, f"/api/jobs/{job_id}/log?offset={offset}")
        payload = json.loads(body)
        if payload.get("text"):
            offset = payload["offset"]
            seen += payload["text"]

        st, body = get(base, f"/api/jobs/{job_id}")
        final = json.loads(body)["job"]
        if final["status"] in ("done", "failed", "cancelled", "interrupted"):
            break
        time.sleep(2)

    check("log streamed incrementally", len(seen) > 500, f"{len(seen)} chars")
    check("log shows the pipeline banner", "VIDEO DUBBER" in seen)
    check("log shows stage 2", "STAGE 2/7" in seen)
    check("pipeline reported the detected hardware", "hardware   :" in seen,
          next((l.strip() for l in seen.splitlines() if "hardware   :" in l), ""))
    check("pipeline reported the auto-chosen models", "models     :" in seen,
          next((l.strip() for l in seen.splitlines() if "models     :" in l), ""))
    check("job reached a terminal state", final is not None
          and final["status"] in ("done", "failed", "cancelled"),
          final["status"] if final else "?")
    if final:
        check("job succeeded", final["status"] == "done", final.get("error") or "")
        prog = final.get("progress", {})
        check("progress was parsed from the log", bool(prog.get("stage_label")),
              f"stage={prog.get('stage_number')} pct={prog.get('percent')}")

    # ---- the run is resumable / cache visible ----------------------------
    work = ROOT / "work" / f"{Path(args.video).stem}_pt"
    check("pipeline wrote its transcript", (work / "transcript.json").exists())

    return _summary()


def _summary() -> int:
    print()
    bad = [r for r in results if not r[1]]
    print(f"  {len(results) - len(bad)}/{len(results)} checks passed")
    print("  RESULT:", "ALL PASSED" if not bad else "FAILURES PRESENT")
    print()
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
