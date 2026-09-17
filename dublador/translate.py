"""Segment-aligned translation.

Translation is done one segment at a time (never on joined batches) because the
dubber needs a strict 1:1 mapping between an original utterance and its
translation -- a merged or re-ordered batch would destroy the timing.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from .config import NON_LATIN, LANGUAGES, normalize_lang
from .schema import Transcript
from .utils import LOG, Progress, banner, ensure_dir, read_json, write_json

#: internal code -> deep-translator / Google code (a few differ)
GOOGLE_CODE: Dict[str, str] = {
    "zh": "zh-CN",
    "he": "iw",
    "no": "no",
    "pt": "pt",
}

BACKENDS = ("google", "mymemory", "libre")

#: MyMemory speaks locale-style codes ("pt-BR"), not bare ISO-639-1 ("pt").
#: Handing it a bare code makes deep-translator raise, which is exactly how the
#: fallback chain used to fail silently on this machine.
MYMEMORY_LOCALE: Dict[str, str] = {
    "en": "en-GB", "pt": "pt-BR", "es": "es-ES", "fr": "fr-FR", "de": "de-DE",
    "it": "it-IT", "nl": "nl-NL", "pl": "pl-PL", "ru": "ru-RU", "ja": "ja-JP",
    "ko": "ko-KR", "zh": "zh-CN", "ar": "ar-SA", "tr": "tr-TR", "sv": "sv-SE",
    "da": "da-DK", "fi": "fi-FI", "no": "nb-NO", "cs": "cs-CZ", "el": "el-GR",
    "he": "he-IL", "hi": "hi-IN", "id": "id-ID", "ro": "ro-RO", "hu": "hu-HU",
    "uk": "uk-UA", "vi": "vi-VN", "th": "th-TH", "bg": "bg-BG", "sk": "sk-SK",
    "sl": "sl-SI", "hr": "hr-HR", "lt": "lt-LT", "lv": "lv-LV", "et": "et-EE",
    "ca": "ca-ES", "fa": "fa-IR", "ms": "ms-MY", "ta": "ta-IN", "bn": "bn-IN",
    "sr": "sr-RS", "ur": "ur-PK", "af": "af-ZA",
}

_MYMEMORY_URL = "https://api.mymemory.translated.net/get"


def mymemory_translate(text: str, source: str, target: str, *,
                       timeout: float = 30.0) -> str:
    """Query the MyMemory REST API directly with ``urllib``.

    This is preferred over ``deep_translator.MyMemoryTranslator`` because it
    takes the locale codes we already know how to build, needs no third-party
    parsing, and is verified working on this restricted network.
    """
    import json as _json  # noqa: PLC0415
    import urllib.parse as _parse  # noqa: PLC0415
    import urllib.request as _request  # noqa: PLC0415

    src = MYMEMORY_LOCALE.get((source or "").lower(), source or "en-GB")
    tgt = MYMEMORY_LOCALE.get((target or "").lower(), target)

    query = _parse.urlencode({"q": text, "langpair": f"{src}|{tgt}"})
    req = _request.Request(f"{_MYMEMORY_URL}?{query}",
                           headers={"User-Agent": "dublador/1.0 (+translate)"})
    with _request.urlopen(req, timeout=timeout) as resp:
        payload = _json.loads(resp.read().decode("utf-8", "replace"))

    status = int(payload.get("responseStatus") or 0)
    if status != 200:
        raise RuntimeError(
            f"MyMemory status {status}: {payload.get('responseDetails')}"
        )

    out = str((payload.get("responseData") or {}).get("translatedText") or "").strip()
    if not out:
        raise RuntimeError("MyMemory returned an empty translation")
    # MyMemory sometimes echoes a "PLEASE SELECT TWO DISTINCT LANGUAGES" notice.
    if out.isupper() and "MYMEMORY WARNING" in out:
        raise RuntimeError(f"MyMemory warning: {out}")
    return out


def _backend_translator(backend: str, source: str, target: str):
    """Build a deep-translator object, importing lazily so the package is optional."""
    try:
        import deep_translator  # noqa: F401,PLC0415
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "deep-translator is not installed; run: python -m pip install deep-translator"
        ) from e

    src = GOOGLE_CODE.get(source, source)
    tgt = GOOGLE_CODE.get(target, target)

    if backend == "google":
        from deep_translator import GoogleTranslator  # noqa: PLC0415

        return GoogleTranslator(source=src or "auto", target=tgt)
    if backend == "mymemory":
        from deep_translator import MyMemoryTranslator  # noqa: PLC0415

        return MyMemoryTranslator(
            source=MYMEMORY_LOCALE.get((source or "").lower(), src or "en-GB"),
            target=MYMEMORY_LOCALE.get((target or "").lower(), tgt),
        )
    if backend == "libre":
        from deep_translator import LibreTranslateTranslator  # noqa: PLC0415

        return LibreTranslateTranslator(source=src or "auto", target=tgt)
    raise ValueError(f"unknown translator backend {backend!r}; pick one of {BACKENDS}")


class TranslationCache:
    """On-disk memo so a re-run costs no network calls."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self.data: Dict[str, str] = read_json(self.path, default={}) or {}
        self.dirty = False

    @staticmethod
    def key(text: str, source: str, target: str, backend: str) -> str:
        raw = f"{backend}|{source}|{target}|{text}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:32]

    def get(self, text: str, source: str, target: str, backend: str) -> Optional[str]:
        with self._lock:
            return self.data.get(self.key(text, source, target, backend))

    def put(self, text: str, source: str, target: str, backend: str, out: str) -> None:
        with self._lock:
            self.data[self.key(text, source, target, backend)] = out
            self.dirty = True

    def save(self) -> None:
        with self._lock:
            if self.dirty:
                write_json(self.path, self.data)
                self.dirty = False

    def __len__(self) -> int:
        return len(self.data)


class TextTranslator:
    """Thread-safe translator with caching, retries and backend fallback."""

    def __init__(
        self,
        source: str,
        target: str,
        *,
        backend: str = "google",
        cache: Optional[TranslationCache] = None,
        workers: int = 4,
        retries: int = 3,
        pause: float = 0.0,
        glossary: Optional[Dict[str, str]] = None,
    ):
        self.source = (source or "auto").lower()
        self.target = normalize_lang(target)
        self.backend = backend
        self.backends = [backend] + [b for b in BACKENDS if b != backend]
        self.cache = cache
        self.workers = max(1, workers)
        self.retries = max(1, retries)
        self.pause = pause
        self.glossary = glossary or {}

        self._local = threading.local()
        self._warned: set = set()
        self.stats = {"hit": 0, "api": 0, "failed": 0}

    # -- internals ---------------------------------------------------------
    def _engine(self, backend: str):
        store = getattr(self._local, "engines", None)
        if store is None:
            store = {}
            self._local.engines = store
        if backend not in store:
            store[backend] = _backend_translator(backend, self.source, self.target)
        return store[backend]

    def _invoke(self, backend: str, text: str) -> str:
        """Call one backend, preferring our own verified REST client."""
        if backend == "mymemory":
            try:
                return mymemory_translate(text, self.source, self.target)
            except Exception:  # noqa: BLE001
                return self._engine("mymemory").translate(text)
        return self._engine(backend).translate(text)

    def _protect(self, text: str) -> str:
        return text

    def _restore(self, text: str) -> str:
        for src, dst in self.glossary.items():
            text = text.replace(src, dst)
        return text

    def _one(self, text: str) -> str:
        text = (text or "").strip()
        if not text:
            return ""

        if self.cache:
            hit = self.cache.get(text, self.source, self.target, self.backend)
            if hit is not None:
                self.stats["hit"] += 1
                return hit

        last_err: Optional[Exception] = None
        for backend in self.backends:
            for attempt in range(self.retries):
                try:
                    out = self._invoke(backend, text)
                    if out and str(out).strip():
                        result = self._restore(str(out).strip())
                        if self.cache:
                            self.cache.put(text, self.source, self.target, self.backend, result)
                        self.stats["api"] += 1
                        if self.pause:
                            time.sleep(self.pause)
                        return result
                    last_err = RuntimeError(f"empty translation from {backend}")
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if backend not in self._warned:
                        self._warned.add(backend)
                        LOG.warning("translator backend '%s' is failing (%s: %s); "
                                    "falling back to the next one",
                                    backend, type(e).__name__, e)
                    time.sleep(min(4.0, 0.6 * (2 ** attempt)))
            LOG.debug("translator backend %s failed: %s", backend, last_err)

        self.stats["failed"] += 1
        LOG.warning("translation failed, keeping source text: %.60s...", text)
        return text

    def translate(self, text: str) -> str:
        return self._one(text)

    def translate_many(self, texts: Sequence[str], *, label: str = "translating") -> List[str]:
        results: List[Optional[str]] = [None] * len(texts)
        prog = Progress(len(texts), label=label, every=max(1, len(texts) // 20 or 1))

        def work(idx: int) -> tuple[int, str]:
            return idx, self._one(texts[idx])

        if self.workers == 1:
            for i in range(len(texts)):
                _, val = work(i)
                results[i] = val
                prog.step()
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                futs = [pool.submit(work, i) for i in range(len(texts))]
                for fut in as_completed(futs):
                    idx, val = fut.result()
                    results[idx] = val
                    prog.step()

        return [r if r is not None else texts[i] for i, r in enumerate(results)]


def translate_transcript(
    tr: Transcript,
    target_lang: str,
    *,
    source_lang: Optional[str] = None,
    backend: str = "google",
    cache_path: Optional[Path] = None,
    workers: int = 4,
    glossary: Optional[Dict[str, str]] = None,
    skip_empty: bool = True,
) -> Transcript:
    """Fill ``segment.translation`` for every segment, in place and concurrently."""
    banner(f"TRANSLATION -> {target_lang}")

    src = (source_lang or tr.language or "auto").lower()
    tgt = normalize_lang(target_lang)

    if src != "auto" and normalize_lang(src) == tgt:
        LOG.info("source and target language are both '%s' - skipping translation", tgt)
        for seg in tr.segments:
            seg.translation = seg.text
        return tr

    cache = TranslationCache(cache_path) if cache_path else None
    translator = TextTranslator(src, tgt, backend=backend, cache=cache,
                                workers=workers, glossary=glossary)

    texts = [seg.text for seg in tr.segments]
    if skip_empty:
        todo_idx = [i for i, t in enumerate(texts) if t.strip()]
    else:
        todo_idx = list(range(len(texts)))

    LOG.info("translating %d segments %s -> %s via %s",
             len(todo_idx), LANGUAGES.get(src, src), LANGUAGES.get(tgt, tgt), backend)

    out_vals = translator.translate_many([texts[i] for i in todo_idx], label="  translated")

    for slot, idx in enumerate(todo_idx):
        tr.segments[idx].translation = out_vals[slot]
    for i in range(len(texts)):
        if tr.segments[i].translation is None:
            tr.segments[i].translation = texts[i]

    if cache:
        cache.save()
        LOG.info("translation cache: %d entries (%d reused, %d fetched, %d failed)",
                 len(cache), translator.stats["hit"], translator.stats["api"],
                 translator.stats["failed"])

    tr.extra["translation"] = {
        "source": src, "target": tgt, "backend": backend, **translator.stats,
    }
    LOG.info("translation done")
    return tr


def translation_needs_romanization(lang: str) -> bool:
    """F5-TTS is trained mostly on Latin script; CJK/Cyrillic/Arabic need care."""
    try:
        return normalize_lang(lang) in NON_LATIN
    except ValueError:
        return False
