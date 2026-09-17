"""End-to-end video dubbing pipeline.

     video ──► extract audio ──► transcribe (Whisper)
           ──► analyse voices (diarize + pick reference clips)
           ──► translate
           ──► clone voices & synthesise (F5-TTS)
           ──► fit each line to its original time slot
           ──► rebuild the audio timeline
           ──► re-animate lips (Wav2Lip) and mux
           ──► dubbed video

Every stage is checkpointed on disk, so a crash or a Ctrl-C on this slow
hardware never costs more than the stage that was running.
"""
from __future__ import annotations

import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import media
from .assemble import build_dub_track, extract_background_bed, separate_vocals_demucs
from .config import (DubladorConfig, INPUT_DIR, LANGUAGES, MODELS_DIR, OUTPUT_DIR,
                     WORK_DIR, ensure_dirs, normalize_lang)
from .schema import Transcript
from .utils import (LOG, StageCache, banner, ensure_dir, file_hash, free_disk_gb,
                    human_duration, read_json, unique_path, write_json)

__version__ = "1.0.0"


@dataclass
class PipelineResult:
    transcript: Transcript
    work_dir: Path
    output_video: Optional[Path] = None
    manifest: Dict[str, Any] = field(default_factory=dict)
    elapsed: float = 0.0
    stopped_after: Optional[str] = None

    def __str__(self) -> str:
        if self.output_video is None:
            return f"(stopped after stage '{self.stopped_after}')  {human_duration(self.elapsed)}"
        return f"{self.output_video}  ({human_duration(self.elapsed)})"


#: Ordered stages accepted by ``--until``.  On slow hardware it is often useful
#: to run the pipeline only as far as translation and inspect the result before
#: committing an hour to synthesis.
STAGE_ORDER = ("extract", "transcribe", "voices", "translate", "tts", "assemble", "finish")


def _stage_index(stage: str) -> int:
    try:
        return STAGE_ORDER.index(stage)
    except ValueError as e:
        raise ValueError(
            f"unknown stage {stage!r}; choose from {', '.join(STAGE_ORDER)}"
        ) from e


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _lipsync_module():
    """Import the lip-sync backend lazily so a broken/absent optional dep
    never stops the rest of the pipeline from working."""
    try:
        from . import lipsync  # noqa: PLC0415

        return lipsync
    except Exception as e:  # noqa: BLE001
        LOG.warning("lip-sync module unavailable (%s); the dub will be muxed "
                    "onto the original picture without mouth re-animation", e)
        return None


def _preflight(cfg: DubladorConfig) -> Path:
    banner("PRE-FLIGHT CHECKS")

    if not media.check_ffmpeg():
        raise RuntimeError("ffmpeg is missing. Run: python tools/bootstrap.py")

    src = Path(cfg.input_video)
    if not src.exists():
        # try the input/ folder by name
        cand = INPUT_DIR / cfg.input_video
        if cand.exists():
            src = cand
        else:
            raise FileNotFoundError(f"input video not found: {cfg.input_video}")
    cfg.input_video = str(src.resolve())

    cfg.target_lang = normalize_lang(cfg.target_lang)
    if cfg.source_lang:
        cfg.source_lang = normalize_lang(cfg.source_lang)

    # ---- self-tune from the detected hardware ---------------------------
    from .hardware import apply_to_config, detect, recommend, report_lines  # noqa: PLC0415

    hw = detect(deep=True)
    rec = recommend(hw)
    if _wants_auto(cfg):
        LOG.info("")
        for line in report_lines(hw, rec):
            LOG.info("  %s", line)
        LOG.info("")
        apply_to_config(cfg)

    info = media.media_info(src)
    if not info.has_video:
        raise RuntimeError(f"{src.name} has no video stream")
    if not info.has_audio:
        raise RuntimeError(
            f"{src.name} has no audio stream - there is nothing to transcribe or dub"
        )

    LOG.info("input      : %s", src.name)
    LOG.info("resolution : %s @ %.2f fps, %.1f MB", info.resolution, info.fps,
             src.stat().st_size / 1048576)
    LOG.info("duration   : %s", human_duration(info.duration))
    LOG.info("audio      : %s, %d ch, %d Hz", info.audio_codec,
             info.audio_channels, info.audio_sample_rate)
    LOG.info("language   : %s -> %s", cfg.source_lang or "auto-detect",
             f"{LANGUAGES.get(cfg.target_lang, cfg.target_lang)} ({cfg.target_lang})")
    LOG.info("hardware   : %s  [%s]", hw.summary, rec.tier_label)
    LOG.info("device     : %s, %d threads", cfg.resolve_device(), cfg.resolve_threads())
    LOG.info("models     : Whisper %s/%s, F5-TTS %s @ %d steps",
             cfg.whisper_model, cfg.whisper_compute_type, cfg.f5_model, cfg.tts_nfe_step)
    LOG.info("disk free  : %.1f GB", free_disk_gb(MODELS_DIR))
    return src


def _wants_auto(cfg: DubladorConfig) -> bool:
    """True when any tunable is still on its 'decide for me' sentinel."""
    return (cfg.device in ("auto", "", None)
            or not cfg.threads
            or cfg.whisper_model in ("auto", "", None)
            or cfg.whisper_compute_type in ("auto", "", None)
            or not cfg.tts_nfe_step)


def _default_output(cfg: DubladorConfig, src: Path) -> Path:
    if cfg.output_video:
        return Path(cfg.output_video)
    ensure_dir(OUTPUT_DIR)
    return OUTPUT_DIR / f"{src.stem}_{cfg.target_lang}_dubbed.mp4"


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------
def _stage_extract_audio(src: Path, work: Path) -> Dict[str, Path]:
    banner("STAGE 1/7  EXTRACT AUDIO")
    out = {
        "asr_16k": work / "audio_16k.wav",       # Whisper + voice analysis
        "voice_24k": work / "audio_24k.wav",     # voice cloning source
        "original": work / "original_24k.wav",   # background bed (optional)
    }
    if not (out["asr_16k"].exists() and out["asr_16k"].stat().st_size > 1024):
        media.extract_audio(src, out["asr_16k"], sample_rate=16000, channels=1)
        LOG.info("  16 kHz mono  -> %s", out["asr_16k"].name)
    else:
        LOG.info("  16 kHz mono  -> cached")

    if not (out["voice_24k"].exists() and out["voice_24k"].stat().st_size > 1024):
        media.extract_audio(src, out["voice_24k"], sample_rate=24000, channels=1)
        LOG.info("  24 kHz mono  -> %s", out["voice_24k"].name)
    else:
        LOG.info("  24 kHz mono  -> cached")

    if not (out["original"].exists() and out["original"].stat().st_size > 1024):
        media.extract_audio(src, out["original"], sample_rate=24000, channels=1)
    return out


def _stage_transcribe(cfg: DubladorConfig, paths: Dict[str, Path], work: Path,
                      cache: StageCache) -> Transcript:
    banner("STAGE 2/7  SPEECH TO TEXT")
    from .transcribe import split_long_segments, transcribe_cached

    json_out = work / "transcript.json"
    if not cache.hit("transcribe", inputs=[paths["asr_16k"]], outputs=[json_out]):
        tr = transcribe_cached(
            paths["asr_16k"], json_out,
            model_size=cfg.whisper_model,
            language=cfg.source_lang,
            device=cfg.resolve_device(),
            compute_type=cfg.whisper_compute_type,
            beam_size=cfg.whisper_beam_size,
            vad_filter=cfg.vad_filter,
            threads=cfg.resolve_threads(),
        )
    else:
        data = read_json(json_out) or {}
        tr = Transcript.from_dict(data)

    if not tr.segments:
        raise RuntimeError(
            "no speech was detected in this video. Nothing to dub. "
            "(If you are sure there is speech, try --whisper-model medium and "
            "disable the VAD filter with --no-vad.)"
        )

    before = len(tr.segments)
    split_long_segments(tr, max_seconds=12.0)
    if len(tr.segments) != before:
        LOG.info("split long lines: %d -> %d segments", before, len(tr.segments))
        write_json(json_out, tr.to_dict())
    return tr


def _stage_voices(cfg: DubladorConfig, tr: Transcript, paths: Dict[str, Path],
                  work: Path, cache: StageCache) -> Transcript:
    banner("STAGE 3/7  VOICE ANALYSIS")
    from .speakers import diarize, select_reference_clips

    refs_dir = ensure_dir(work / "refs")
    done = all(Path(p.ref_audio).exists() for p in tr.speakers.values() if p.ref_audio)
    if not (cfg.resume and tr.speakers and done):
        tr = diarize(
            paths["asr_16k"], tr,
            device=cfg.resolve_device(),
            threshold=cfg.speaker_threshold,
            min_speakers=cfg.min_speakers,
            max_speakers=cfg.max_speakers,
        )
        tr = select_reference_clips(
            paths["asr_16k"], tr,
            ref_min=cfg.ref_min_seconds, ref_max=cfg.ref_max_seconds,
            out_dir=refs_dir,
        )
        write_json(work / "transcript_voices.json", tr.to_dict())
    else:
        LOG.info("  reusing previously detected speakers and reference clips")
    return tr


def _stage_translate(cfg: DubladorConfig, tr: Transcript, work: Path) -> Transcript:
    banner("STAGE 4/7  TRANSLATION")
    from .translate import translate_transcript

    cache_path = work / "translation_cache.json"
    tr = translate_transcript(
        tr, cfg.target_lang,
        source_lang=cfg.source_lang or tr.language,
        backend=cfg.translator,
        cache_path=cache_path,
        workers=max(1, min(8, cfg.resolve_threads() * 2)),
    )
    write_json(work / "transcript_translated.json", tr.to_dict())
    (work / "subtitles_target.srt").write_text(tr.to_srt(translated=True), encoding="utf-8")
    (work / "subtitles_source.srt").write_text(tr.to_srt(translated=False), encoding="utf-8")
    LOG.info("  wrote subtitles_target.srt and subtitles_source.srt")
    return tr


def _stage_tts(cfg: DubladorConfig, tr: Transcript, paths: Dict[str, Path],
               work: Path, cache: StageCache) -> Transcript:
    banner("STAGE 5/7  VOICE CLONING AND SYNTHESIS")
    from .tts import f5_import_error, synthesize_transcript

    err = f5_import_error()
    if err:
        raise RuntimeError(
            f"F5-TTS is not usable in this environment: {err}\n"
            "Install it with: python -m pip install f5-tts  (then provision its "
            "inference deps; see README). The first synthesis also downloads "
            "~1.4 GB of model weights."
        )

    tr = synthesize_transcript(
        tr, paths["asr_16k"], work / "tts", cfg,
        cache=cache, fitted_dir=work / "tts" / "fitted",
    )
    write_json(work / "transcript_tts.json", tr.to_dict())
    return tr


def _stage_assemble(cfg: DubladorConfig, tr: Transcript, paths: Dict[str, Path],
                    work: Path, duration: float) -> Dict[str, Any]:
    banner("STAGE 6/7  REBUILD AUDIO TIMELINE")

    dub_raw = work / "dub_track.wav"
    dub, meta = build_dub_track(tr, duration, dub_raw, gap=cfg.segment_gap)

    final_audio = dub
    used_background = False

    if cfg.background_mode in ("duck", "separate"):
        bed: Optional[Path] = None
        if cfg.background_mode == "separate":
            bed = separate_vocals_demucs(paths["original"], ensure_dir(work / "demucs"))
            if bed is None:
                LOG.info("  falling back to 'duck' mode")
        if bed is None:
            bed = paths["original"]

        if bed and Path(bed).exists():
            mixed = work / "dub_mixed.wav"
            media.mix_audio(
                dub, bed, mixed,
                primary_db=cfg.voice_gain_db,
                secondary_db=cfg.background_gain_db,
                duck=(cfg.background_mode == "duck"),
                duration=duration if duration else None,
            )
            final_audio = mixed
            used_background = True
            LOG.info("  mixed dub with the original bed at %.1f dB", cfg.background_gain_db)
    else:
        LOG.info("  background_mode=none -> delivering a clean dub "
                 "(original soundtrack discarded)")

    meta["background"] = {"mode": cfg.background_mode, "applied": used_background}
    meta["final_audio"] = str(final_audio)
    return meta


def _stage_finish(cfg: DubladorConfig, src: Path, tr: Transcript,
                  dub_meta: Dict[str, Any], work: Path, out_path: Path,
                  duration: float = 0.0) -> Path:
    banner("STAGE 7/7  LIP SYNC AND MUX")

    audio = Path(dub_meta["final_audio"])
    ensure_dir(out_path.parent)

    # ---- lip sync --------------------------------------------------------
    module = _lipsync_module()
    synced: Optional[Path] = None

    # Wav2Lip costs ~1 s/frame here (face detection dominates), so an
    # hour-long video would take days. Refuse by default and explain why.
    fps = media.media_info(src).fps or 25.0
    est_seconds = duration * fps if duration else 0.0

    if not cfg.lipsync:
        LOG.info("  lip sync disabled")
    elif module is None:
        LOG.info("  lip-sync module not present")
    elif not module.lipsync_available():
        LOG.warning("  Wav2Lip backend is not installed - skipping lip sync")
    elif est_seconds > cfg.lipsync_max_seconds * fps and not cfg.lipsync_force:
        LOG.warning("  skipping lip sync: %.0fs of video at %.0f fps is about %.0f frames, "
                    "and Wav2Lip needs roughly 1 s per frame on this CPU "
                    "(approximately %s of pure compute).",
                    duration, fps, est_seconds, human_duration(est_seconds))
        LOG.warning("  the dubbed audio will be muxed onto the original picture instead.")
        LOG.warning("  override with --lipsync-force, raise the limit with "
                    "--lipsync-max-seconds, or speed it up with --lipsync-resize 2.")
    else:
        tmp_out = work / "lipsynced.mp4"
        try:
            LOG.info("  running Wav2Lip on CPU (~1 s/frame; about %s expected)",
                     human_duration(est_seconds))
            synced = module.lipsync_video(
                src, audio, tmp_out,
                device=cfg.resolve_device(),
                batch_size=cfg.lipsync_batch,
                resize_factor=cfg.lipsync_resize_factor,
            )
            LOG.info("  lip sync done -> %s", Path(synced).name)
        except Exception as e:  # noqa: BLE001
            LOG.error("  lip sync failed (%s); continuing without it", e)
            synced = None

    # ---- mux -------------------------------------------------------------
    if synced and Path(synced).exists():
        LOG.info("  using the lip-synced picture")
        if Path(synced) != out_path:
            shutil.copy2(synced, out_path)
    else:
        LOG.info("  muxing the dubbed audio onto the original picture")
        media.mux_video_audio(src, audio, out_path, copy_video=False)

    size = out_path.stat().st_size / 1048576
    LOG.info("  output: %s (%.1f MB)", out_path, size)
    return out_path


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def run_pipeline(cfg: DubladorConfig, until: str = "all") -> PipelineResult:
    t0 = time.time()
    ensure_dirs()

    stop_at = None if until in ("all", "", None) else _stage_index(until)

    banner(f"VIDEO DUBBER v{__version__}")
    if stop_at is not None:
        LOG.info("will stop after stage: %s", until)
    src = _preflight(cfg)

    if not cfg.job_name or cfg.job_name == "job":
        cfg.job_name = f"{src.stem}_{cfg.target_lang}"
    work = ensure_dir(WORK_DIR / cfg.job_name)
    LOG.info("work dir   : %s", work)

    cfg.save(work / "config.json")
    cache = StageCache(work, enabled=cfg.resume)

    info = media.media_info(src)
    duration = info.duration

    def partial(last: str, transcript: Transcript) -> PipelineResult:
        LOG.info("")
        LOG.info("stopping after '%s' as requested (--until %s)", last, until)
        LOG.info("re-run without --until to continue; completed stages are cached.")
        return PipelineResult(
            transcript=transcript, work_dir=work, output_video=None,
            manifest={"stopped_after": last, "work_dir": str(work)},
            elapsed=time.time() - t0, stopped_after=last,
        )

    def reached(stage: str) -> bool:
        return stop_at is not None and _stage_index(stage) >= stop_at

    # 1 -------------------------------------------------------------------
    paths = _stage_extract_audio(src, work)
    if reached("extract"):
        return partial("extract", Transcript())

    # 2 -------------------------------------------------------------------
    tr = _stage_transcribe(cfg, paths, work, cache)
    if reached("transcribe"):
        return partial("transcribe", tr)

    # 3 -------------------------------------------------------------------
    tr = _stage_voices(cfg, tr, paths, work, cache)
    if reached("voices"):
        return partial("voices", tr)

    # 4 -------------------------------------------------------------------
    tr = _stage_translate(cfg, tr, work)
    if reached("translate"):
        return partial("translate", tr)

    # 5 -------------------------------------------------------------------
    tr = _stage_tts(cfg, tr, paths, work, cache)
    if reached("tts"):
        return partial("tts", tr)

    # 6 -------------------------------------------------------------------
    dub_meta = _stage_assemble(cfg, tr, paths, work, duration)
    if reached("assemble"):
        return partial("assemble", tr)

    # 7 -------------------------------------------------------------------
    out_path = _default_output(cfg, src)
    out_path = _stage_finish(cfg, src, tr, dub_meta, work, out_path, duration)

    elapsed = time.time() - t0
    manifest = {
        "version": __version__,
        "input": str(src),
        "output": str(out_path),
        "target_language": cfg.target_lang,
        "source_language": cfg.source_lang or tr.language,
        "duration_seconds": round(duration, 3),
        "segments": len(tr.segments),
        "speakers": {
            k: {
                "segments": len(v.segments),
                "speech_seconds": round(v.total_speech, 2),
                "pitch_hz": round(v.mean_pitch_hz, 1),
                "gender": v.gender,
                "reference": Path(v.ref_audio).name if v.ref_audio else None,
            }
            for k, v in tr.speakers.items()
        },
        "tts": tr.extra.get("tts", {}),
        "dub_track": {k: v for k, v in dub_meta.items() if k != "final_audio"},
        "elapsed_seconds": round(elapsed, 2),
        "config": cfg.to_dict(),
    }
    write_json(work / "manifest.json", manifest)

    banner("DONE")
    LOG.info("dubbed video : %s", out_path)
    LOG.info("subtitles    : %s", work / "subtitles_target.srt")
    LOG.info("work dir     : %s", work)
    LOG.info("total time   : %s", human_duration(elapsed))
    if tr.extra.get("tts"):
        LOG.info("speech time  : %.1fs generated (realtime factor %.1fx)",
                 tr.extra["tts"].get("audio_seconds", 0.0),
                 tr.extra["tts"].get("realtime_factor", 0.0))

    return PipelineResult(output_video=out_path, transcript=tr, work_dir=work,
                          manifest=manifest, elapsed=elapsed)


def run_transcribe_only(cfg: DubladorConfig) -> Dict[str, Any]:
    """Cheap smoke test: probe + extract + transcribe, no models beyond Whisper."""
    ensure_dirs()
    banner("TRANSCRIBE-ONLY MODE")
    src = _preflight(cfg)
    if not cfg.job_name or cfg.job_name == "job":
        cfg.job_name = f"{src.stem}_probe"
    work = ensure_dir(WORK_DIR / cfg.job_name)
    paths = _stage_extract_audio(src, work)
    cache = StageCache(work, enabled=cfg.resume)
    tr = _stage_transcribe(cfg, paths, work, cache)
    return {
        "language": tr.language,
        "segments": len(tr.segments),
        "duration": tr.duration,
        "text": " ".join(s.text for s in tr.segments)[:1500],
        "transcript_json": str(work / "transcript.json"),
    }
