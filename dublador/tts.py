"""Voice cloning text-to-speech with F5-TTS.

F5-TTS is a flow-matching TTS model that clones a voice from a short reference
clip plus its transcript -- no per-speaker fine-tuning needed.  For dubbing we
exploit three properties:

1. **Voice cloning** -- each detected speaker becomes an ``F5TTS`` reference
   pair (``ref_file`` + ``ref_text``), so every character keeps its own voice.
2. **``fix_duration``** -- F5-TTS can be told to generate speech that occupies
   an exact number of seconds.  Passing the original utterance duration makes
   the dub land in the right slot with no time-stretching artefacts.
3. **``speed``** -- a secondary lever when ``fix_duration`` alone is not
   enough to fit a long translation into a short slot.

Everything is CPU-capable, but on this i3-4130 expect roughly 10-40x slower
than realtime; ``nfe_step`` and ``tts_fix_duration`` are the main cost knobs.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .compat import install_torchaudio_shims
from .config import MODELS_DIR, DubladorConfig
from .media import audio_duration, convert_audio, fit_to_slot
from .schema import Segment, SpeakerProfile, Transcript
from .utils import (LOG, Progress, StageCache, banner, ensure_dir, free_disk_gb,
                    human_duration, read_json, write_json)

#: vocos-mel-24khz is what F5-TTS's default vocoder emits.
SAMPLE_RATE = 24000

_MODEL_CACHE: Dict[str, Any] = {}


# --------------------------------------------------------------------------
# availability / loading
# --------------------------------------------------------------------------
def f5_available() -> bool:
    try:
        install_torchaudio_shims()
        import f5_tts.api  # noqa: F401,PLC0415

        return True
    except Exception as e:  # noqa: BLE001
        LOG.debug("F5-TTS unavailable: %s", e)
        return False


def f5_import_error() -> Optional[str]:
    try:
        install_torchaudio_shims()
        import f5_tts.api  # noqa: F401,PLC0415

        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


class _QuietTqdm:
    """Stand-in for the ``tqdm`` module F5-TTS expects as ``progress=``."""

    enabled = False

    @staticmethod
    def tqdm(iterable=None, **kwargs):  # noqa: ANN001
        if _QuietTqdm.enabled:
            import tqdm as _t  # noqa: PLC0415

            return _t.tqdm(iterable, **kwargs)
        return iterable if iterable is not None else []


def resolve_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch  # noqa: PLC0415

        if torch.cuda.is_available():
            return "cuda"
    except Exception:  # noqa: BLE001
        pass
    return "cpu"


def load_f5(cfg: DubladorConfig):
    """Load (and memoise) the F5-TTS model + vocoder."""
    install_torchaudio_shims()

    device = resolve_device(cfg.device)
    key = f"{cfg.f5_model}|{cfg.f5_ckpt}|{device}"
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    err = f5_import_error()
    if err:
        raise RuntimeError(f"F5-TTS is not usable in this environment: {err}")

    from f5_tts.api import F5TTS  # noqa: PLC0415

    hf_cache = ensure_dir(MODELS_DIR / "hf")
    _QuietTqdm.enabled = bool(cfg.tts_show_progress)

    approx = 1.35 if "v1_Base" in cfg.f5_model or cfg.f5_model == "F5TTS_Base" else 0.45
    cache_has_model = any(hf_cache.rglob("*.safetensors")) or any(hf_cache.rglob("*.pt"))
    if not cache_has_model:
        free = free_disk_gb(MODELS_DIR)
        LOG.info("  first run: F5-TTS will download ~%.2f GB of weights to %s", approx, hf_cache)
        if free == free and free < approx + 1.0:  # NaN-safe
            LOG.warning("  only %.1f GB free - this may not fit", free)

    LOG.info("loading F5-TTS '%s' on %s (this can take a minute)", cfg.f5_model, device)

    kwargs: Dict[str, Any] = {
        "model": cfg.f5_model,
        "device": device,
        "hf_cache_dir": str(hf_cache),
    }
    if cfg.f5_ckpt:
        kwargs["ckpt_file"] = cfg.f5_ckpt
    if cfg.f5_vocab:
        kwargs["vocab_file"] = cfg.f5_vocab

    last: Optional[Exception] = None
    for use_ema in (True, False):
        try:
            model = F5TTS(use_ema=use_ema, **kwargs)
            LOG.info("  F5-TTS ready (use_ema=%s, sample rate %s Hz)",
                     use_ema, model.target_sample_rate)
            _MODEL_CACHE[key] = model
            return model
        except Exception as e:  # noqa: BLE001
            last = e
            LOG.debug("F5TTS(use_ema=%s) failed: %s", use_ema, e)

    raise RuntimeError(f"could not load F5-TTS model '{cfg.f5_model}': {last}")


# --------------------------------------------------------------------------
# text preparation
# --------------------------------------------------------------------------
_MD = re.compile(r"[*_`#>\[\]()]|https?://\S+")


def clean_text_for_tts(text: str, *, max_chars: int = 220) -> str:
    """Make a translated line safe and pleasant to synthesise.

    F5-TTS is trained on punctuated prose, so we strip markup, collapse
    whitespace and guarantee terminal punctuation.
    """
    t = (text or "").strip()
    if not t:
        return ""
    t = _MD.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    # Remove stray speaker labels such as "SPK_01:" that may leak in.
    t = re.sub(r"^SPK[_-]?\d+\s*[:\-]\s*", "", t, flags=re.IGNORECASE)
    if t and t[-1] not in ".!?,;:…\"'":
        t += "."
    return t


def split_for_tts(text: str, max_chars: int = 220) -> List[str]:
    """Split an over-long line at sentence, then clause, then word bounds."""
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []

    parts: List[str] = []
    # 1) sentence boundaries
    for chunk in re.split(r"(?<=[.!?…])\s+", text):
        if not chunk.strip():
            continue
        if len(chunk) <= max_chars:
            parts.append(chunk.strip())
            continue
        # 2) clause boundaries
        for sub in re.split(r"(?<=[,;:])\s+", chunk):
            if not sub.strip():
                continue
            if len(sub) <= max_chars:
                parts.append(sub.strip())
                continue
            # 3) hard word wrap
            cur = ""
            for w in sub.split():
                if cur and len(cur) + 1 + len(w) > max_chars:
                    parts.append(cur)
                    cur = w
                else:
                    cur = f"{cur} {w}".strip()
            if cur:
                parts.append(cur)
    return [p for p in parts if p]


# --------------------------------------------------------------------------
# the synthesiser
# --------------------------------------------------------------------------
class VoiceCloner:
    """Synthesise lines with per-speaker cloned voices, resumable and buffered."""

    def __init__(self, cfg: DubladorConfig, out_dir: Path):
        self.cfg = cfg
        self.out_dir = ensure_dir(out_dir)
        self.model = None
        self.device = resolve_device(cfg.device)
        self.total_audio_s = 0.0
        self.total_wall_s = 0.0
        self.n_calls = 0

    def load(self) -> None:
        # Defensive: usable without the pipeline having resolved "auto" first.
        if self.cfg.tts_nfe_step <= 0:
            try:
                from .hardware import recommend  # noqa: PLC0415

                self.cfg.tts_nfe_step = recommend().nfe_step
                LOG.info("  [auto] passos de sintese: %d", self.cfg.tts_nfe_step)
            except Exception:  # noqa: BLE001
                self.cfg.tts_nfe_step = 16
        if self.model is None:
            self.model = load_f5(self.cfg)

    # -- one line ----------------------------------------------------------
    def _infer(
        self,
        gen: str,
        ref_audio: Path,
        ref_text: str,
        out_wav: Path,
        fix_duration: Optional[float],
        seed: Optional[int],
    ) -> Path:
        """Single F5-TTS call. ``fix_duration`` is the TOTAL duration seen by
        F5-TTS (reference + generated); see :meth:`synthesize`."""
        wav, sr, _spec = self.model.infer(  # type: ignore[union-attr]
            ref_file=str(ref_audio),
            ref_text=ref_text,
            gen_text=gen,
            show_info=lambda *a, **k: None,
            progress=_QuietTqdm,
            target_rms=self.cfg.tts_target_rms,
            cross_fade_duration=self.cfg.tts_cross_fade,
            sway_sampling_coef=self.cfg.tts_sway_coef,
            cfg_strength=self.cfg.tts_cfg_strength,
            nfe_step=self.cfg.tts_nfe_step,
            speed=self.cfg.tts_speed,
            fix_duration=fix_duration,
            remove_silence=self.cfg.tts_remove_silence,
            file_wave=str(out_wav),
            file_spec=None,
            seed=self.cfg.seed if seed is None else seed,
        )
        return out_wav

    def synthesize(
        self,
        text: str,
        ref_audio: Path,
        ref_text: str,
        out_wav: Path,
        *,
        target_seconds: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> Path:
        """Synthesise ``text`` in the cloned voice of ``ref_audio``.

        ``target_seconds`` is how long the **generated speech** should be, i.e.
        the slot left by the original line.  F5-TTS's own ``fix_duration``
        parameter instead expects the *total* length **including the reference
        clip** (upstream computes ``target_total = fix_duration - ref_sec``), so
        passing the slot duration directly silently produces empty audio
        whenever the reference is longer than the slot.  We add the reference
        length here and defend against that failure mode.
        """
        self.load()
        out_wav = Path(out_wav)
        ensure_dir(out_wav.parent)

        gen = clean_text_for_tts(text, max_chars=self.cfg.tts_max_chars)
        if not gen:
            raise ValueError("empty text to synthesise")

        # F5-TTS hallucinates when the reference transcript is empty.
        ref_text = (ref_text or "").strip()
        if not ref_text:
            LOG.debug("  reference text missing; letting F5-TTS transcribe it")

        ref_seconds = audio_duration(ref_audio)

        t0 = time.time()
        if target_seconds and target_seconds > 0.1:
            self._infer(gen, ref_audio, ref_text, out_wav,
                        ref_seconds + float(target_seconds), seed)
            dur = audio_duration(out_wav)

            # Guard: if asking for a fixed length swallowed the speech (e.g. a
            # very long reference or an unexpected upstream semantic change),
            # retry letting F5-TTS choose, and let fit_to_slot handle timing.
            if dur < max(0.25, 0.35 * float(target_seconds)):
                LOG.warning("  fix_duration produced only %.2fs for a %.2fs slot "
                            "(reference is %.2fs) - retrying without it",
                            dur, target_seconds, ref_seconds)
                out_wav.unlink(missing_ok=True)
                self._infer(gen, ref_audio, ref_text, out_wav, None, seed)
                dur = audio_duration(out_wav)
        else:
            self._infer(gen, ref_audio, ref_text, out_wav, None, seed)
            dur = audio_duration(out_wav)

        wall = time.time() - t0
        self.n_calls += 1
        self.total_wall_s += wall
        self.total_audio_s += dur
        LOG.debug("  synth %.2fs audio in %.1fs (rtf %.1fx)",
                  dur, wall, (wall / dur) if dur else 0.0)
        return out_wav

    # -- many lines --------------------------------------------------------
    def synthesize_text(
        self,
        text: str,
        ref_audio: Path,
        ref_text: str,
        out_wav: Path,
        *,
        target_seconds: Optional[float] = None,
    ) -> Path:
        """Synthesise arbitrarily long text by splitting and concatenating."""
        pieces = split_for_tts(clean_text_for_tts(text, max_chars=self.cfg.tts_max_chars),
                               self.cfg.tts_max_chars)
        if not pieces:
            raise ValueError("empty text to synthesise")
        if len(pieces) == 1:
            return self.synthesize(pieces[0], ref_audio, ref_text, out_wav,
                                   target_seconds=target_seconds)

        from .media import concat_audio  # noqa: PLC0415

        parts: List[Path] = []
        budget = target_seconds
        total_chars = max(1, sum(len(p) for p in pieces))
        for i, piece in enumerate(pieces):
            part = out_wav.with_name(f"{out_wav.stem}.p{i:03d}{out_wav.suffix}")
            share = None
            if budget:
                # give each piece a slice of the slot proportional to its length
                share = budget * (len(piece) / total_chars)
            self.synthesize(piece, ref_audio, ref_text, part, target_seconds=share)
            parts.append(part)

        concat_audio(parts, out_wav, sample_rate=SAMPLE_RATE, channels=1)
        for p in parts:
            try:
                p.unlink()
            except OSError:
                pass
        return out_wav

    def stats(self) -> Dict[str, Any]:
        rtf = (self.total_wall_s / self.total_audio_s) if self.total_audio_s else 0.0
        return {
            "calls": self.n_calls,
            "audio_seconds": round(self.total_audio_s, 2),
            "wall_seconds": round(self.total_wall_s, 2),
            "realtime_factor": round(rtf, 2),
        }


# --------------------------------------------------------------------------
# transcript-level driver
# --------------------------------------------------------------------------
def synthesize_transcript(
    tr: Transcript,
    audio_16k: Path,
    out_dir: Path,
    cfg: DubladorConfig,
    *,
    cache: Optional[StageCache] = None,
    fitted_dir: Optional[Path] = None,
) -> Transcript:
    """Clone every speaker's voice and render each translated line.

    Writes ``segment.tts_audio`` (raw F5-TTS output) and
    ``segment.fitted_audio`` (time-fitted to the original slot).
    """
    banner("VOICE CLONING + SPEECH SYNTHESIS (F5-TTS)")

    err = f5_import_error()
    if err:
        raise RuntimeError(f"F5-TTS is not usable: {err}")

    out_dir = ensure_dir(out_dir)
    fitted_dir = ensure_dir(fitted_dir or (out_dir / "fitted"))

    # ---- references ------------------------------------------------------
    from .speakers import select_reference_clips  # noqa: PLC0415

    if not tr.speakers or not any(p.ref_audio for p in tr.speakers.values()):
        LOG.info("no reference clips yet - extracting them now")
        tr = select_reference_clips(
            audio_16k, tr,
            ref_min=cfg.ref_min_seconds, ref_max=cfg.ref_max_seconds,
            out_dir=out_dir / "refs",
        )

    usable = {k: p for k, p in tr.speakers.items() if p.ref_audio and Path(p.ref_audio).exists()}
    if not usable:
        raise RuntimeError(
            "no speaker reference clips could be extracted; cannot clone a voice. "
            "Check that the video actually contains speech."
        )

    for spk, prof in sorted(usable.items()):
        LOG.info("  voice %s <- %s (%.1fs reference, pitch %.0f Hz)",
                 spk, Path(prof.ref_audio).name,
                 (prof.ref_end - prof.ref_start), prof.mean_pitch_hz)

    # segments without a speaker fall back to the first voice
    fallback_spk = sorted(usable)[0]
    for seg in tr.segments:
        if not seg.speaker or seg.speaker not in usable:
            seg.speaker = fallback_spk

    cloner = VoiceCloner(cfg, out_dir)
    cloner.load()

    todo = [s for s in tr.segments if s.text_for_tts]
    LOG.info("synthesising %d line(s) with nfe_step=%d, fix_duration=%s, device=%s",
             len(todo), cfg.tts_nfe_step, cfg.tts_fix_duration, cloner.device)
    if not todo:
        LOG.warning("nothing to synthesise")
        return tr

    prog = Progress(len(todo), label="  tts", every=1)
    t_start = time.time()
    failed = 0

    for seg in todo:
        prof = usable[seg.speaker]  # type: ignore[index]
        raw = out_dir / f"seg_{seg.id:05d}.wav"
        fitted = fitted_dir / f"seg_{seg.id:05d}.wav"

        reuse = (cfg.resume and raw.exists() and raw.stat().st_size > 1024
                 and fitted.exists() and fitted.stat().st_size > 1024)
        if not reuse:
            use_fix = cfg.tts_fix_duration and seg.duration >= 0.6
            try:
                cloner.synthesize_text(
                    seg.text_for_tts,
                    Path(prof.ref_audio),  # type: ignore[arg-type]
                    prof.ref_text or "",
                    raw,
                    target_seconds=(seg.duration if use_fix else None),
                )
            except Exception as e:  # noqa: BLE001
                failed += 1
                LOG.error("  segment %d synthesis failed: %s", seg.id, e)
                # leave silence so timing downstream stays intact
                from .media import make_silence  # noqa: PLC0415

                make_silence(raw, max(0.05, seg.duration), sample_rate=SAMPLE_RATE, channels=1)

            fitted_path, meta = fit_to_slot(
                raw, fitted, seg.duration,
                sample_rate=SAMPLE_RATE, channels=1,
                max_speed=cfg.max_stretch, min_speed=cfg.min_stretch,
            )
            meta["speaker"] = seg.speaker
            meta["fix_duration"] = bool(cfg.tts_fix_duration and seg.duration >= 0.6)
            seg.fit_meta = meta
        else:
            meta = read_json(fitted.with_suffix(".json"), default={}) or {}
            if meta:
                seg.fit_meta = meta

        seg.tts_audio = str(raw)
        seg.fitted_audio = str(fitted)
        write_json(fitted.with_suffix(".json"), seg.fit_meta or {})
        prog.step()

    elapsed = time.time() - t_start
    st = cloner.stats()
    LOG.info("synthesis complete in %s (%d calls, %.1fs of speech, realtime factor %.1fx)%s",
             human_duration(elapsed), st["calls"], st["audio_seconds"],
             st["realtime_factor"], f", {failed} failed" if failed else "")

    tr.extra["tts"] = {**st, "failed": failed, "model": cfg.f5_model, "device": cloner.device}
    if cache is not None:
        cache.mark("tts", inputs=[audio_16k, Path(usable[fallback_spk].ref_audio)],  # type: ignore[arg-type]
                   outputs=[Path(s.fitted_audio) for s in tr.segments if s.fitted_audio],
                   model=cfg.f5_model)
    return tr
