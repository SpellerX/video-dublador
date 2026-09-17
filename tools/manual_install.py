#!/usr/bin/env python
"""Install a pure-Python package from an sdist *without* invoking a build backend.

Why this exists
---------------
This host's DSH sandbox denies subprocess stdio pipes.  pip needs pipes to run
PEP 517 build hooks, so **any** sdist install fails with
``WinError 5: Acesso negado while executing command installing build
dependencies``.  Wheels are unaffected.

Some required packages ship sdist-only (e.g. ``antlr4-python3-runtime==4.9.3``,
which ``omegaconf``/``hydra-core`` pin exactly).  For pure-Python packages we
can skip building altogether: download the sdist, extract it, copy the import
packages into ``site-packages`` and synthesise a ``.dist-info`` so that
``importlib.metadata`` -- and therefore pip's dependency resolver -- sees a
normal installation.

Only use this for packages with no compiled extensions.

Usage::

    python tools/manual_install.py antlr4-python3-runtime 4.9.3
    python tools/manual_install.py --url <sdist-url> --name foo --version 1.0
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

SKIP_DIRS = {
    "test", "tests", "testing", "doc", "docs", "example", "examples",
    "build", "dist", "benchmark", "benchmarks", "scripts", ".github",
    "node_modules", "__pycache__", ".git",
}


def site_packages() -> Path:
    for p in sys.path:
        cand = Path(p)
        if cand.name == "site-packages" and cand.is_dir():
            return cand
    raise RuntimeError("could not locate site-packages; run with the project interpreter")


def pypi_sdist_url(name: str, version: str) -> str:
    url = f"https://pypi.org/pypi/{name}/{version}/json"
    with urllib.request.urlopen(url, timeout=60) as r:
        meta = json.load(r)
    for f in meta.get("urls", []):
        if f["filename"].endswith((".tar.gz", ".zip")):
            return f["url"]
    raise SystemExit(f"no sdist found for {name} {version}")


def download(url: str, dest: Path) -> Path:
    print(f"  downloading {url}")
    with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as fh:
        shutil.copyfileobj(r, fh)
    print(f"  saved {dest.name} ({dest.stat().st_size / 1024:.0f} KB)")
    return dest


def extract(archive: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z:
            z.extractall(dest)
    else:
        with tarfile.open(archive) as t:
            t.extractall(dest)
    entries = [p for p in dest.iterdir() if p.is_dir()]
    if len(entries) == 1:
        return entries[0]
    return dest


def find_packages(root: Path) -> list[Path]:
    """Locate importable package directories inside an extracted sdist."""
    found: list[Path] = []

    def scan(base: Path, depth: int = 0) -> None:
        if depth > 3:
            return
        for child in sorted(base.iterdir()):
            if not child.is_dir():
                continue
            if child.name in SKIP_DIRS or child.name.endswith((".egg-info", ".dist-info")):
                continue
            if (child / "__init__.py").exists():
                found.append(child)
            else:
                # a src/ style layout
                if child.name in ("src", "lib", "python"):
                    scan(child, depth + 1)

    scan(root)
    # drop nested packages that are children of an already-found package
    top: list[Path] = []
    for p in found:
        if not any(p != q and q in p.parents for q in found):
            top.append(p)
    return top


def write_dist_info(sp: Path, name: str, version: str) -> Path:
    safe = name.replace("-", "_")
    di = sp / f"{safe}-{version}.dist-info"
    di.mkdir(parents=True, exist_ok=True)
    (di / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
        f"Summary: manually vendored sdist (see tools/manual_install.py)\n",
        encoding="utf-8",
    )
    (di / "WHEEL").write_text(
        "Wheel-Version: 1.0\nGenerator: manual_install.py\n"
        "Root-Is-Purelib: true\nTag: py3-none-any\n",
        encoding="utf-8",
    )
    (di / "INSTALLER").write_text("manual_install.py\n", encoding="utf-8")
    (di / "RECORD").write_text("", encoding="utf-8")
    return di


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    ap.add_argument("version", nargs="?")
    ap.add_argument("--url")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.url and not (args.name and args.version):
        ap.error("give either <name> <version> or --url")

    name = args.name
    version = args.version
    url = args.url or pypi_sdist_url(name, version)
    if not name:
        name = Path(url).name.split("-")[0]

    sp = site_packages()
    print(f"target site-packages: {sp}")

    with tempfile.TemporaryDirectory() as td:
        archive = download(url, Path(td) / Path(url).name)
        root = extract(archive, Path(td) / "x")
        pkgs = find_packages(root)
        if not pkgs:
            print("ERROR: no importable package found in sdist")
            return 1
        print(f"  found packages: {[p.name for p in pkgs]}")
        if args.dry_run:
            return 0
        for pkg in pkgs:
            target = sp / pkg.name
            if target.exists():
                shutil.rmtree(target, ignore_errors=True)
            shutil.copytree(pkg, target, dirs_exist_ok=True)
            print(f"  installed {pkg.name} -> {target}")

    di = write_dist_info(sp, name, version)
    print(f"  wrote metadata {di.name}")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
