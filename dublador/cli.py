"""Command line interface for the video dubber."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import (DubladorConfig, INPUT_DIR, LANGUAGES, MODELS_DIR, OUTPUT_DIR,
                     PROJECT_ROOT, WORK_DIR, ensure_dirs, normalize_lang)
from .utils import LOG, human_duration, setup_logging

PROG = "dublador"


# --------------------------------------------------------------------------
# environment doctor
# --------------------------------------------------------------------------
def environment_report() -> Dict[str, Any]:
    """Check every external dependency and report status without side effects."""
    rep: Dict[str, Any] = {"python": sys.version.split()[0], "checks": {}}

    def check(name: str, fn) -> None:  # noqa: ANN001
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        rep["checks"][name] = {"ok": bool(ok), "detail": detail}

    from . import media  # noqa: PLC0415

    check("ffmpeg", lambda: (media.check_ffmpeg(), str(media.FFMPEG_BIN)))

    def _torch():  # noqa: ANN202
        import torch  # noqa: PLC0415

        return True, f"{torch.__version__} (cuda={torch.cuda.is_available()})"

    def _torchaudio_io():  # noqa: ANN202
        from .compat import install_torchaudio_shims, torchaudio_shim_needed  # noqa: PLC0415

        needed = torchaudio_shim_needed()
        install_torchaudio_shims()
        return True, ("soundfile shim installed (TorchCodec absent)"
                      if needed else "native torchcodec backend")

    def _whisper():  # noqa: ANN202
        import faster_whisper  # noqa: PLC0415

        return True, getattr(faster_whisper, "__version__", "installed")

    def _f5():  # noqa: ANN202
        from .tts import f5_import_error  # noqa: PLC0415

        err = f5_import_error()
        return (err is None), (err or "importable")

    def _translate():  # noqa: ANN202
        import deep_translator  # noqa: PLC0415

        return True, getattr(deep_translator, "__version__", "installed")

    def _speaker():  # noqa: ANN202
        from .speakers import choose_backend  # noqa: PLC0415

        b = choose_backend("auto")
        return True, f"{b} backend"

    def _lipsync():  # noqa: ANN202
        from . import lipsync  # noqa: PLC0415

        return lipsync.lipsync_available(), "Wav2Lip"

    def _disk():  # noqa: ANN202
        from .utils import free_disk_gb  # noqa: PLC0415

        gb = free_disk_gb(PROJECT_ROOT)
        return gb > 4.0, f"{gb:.1f} GB free"

    check("torch", _torch)
    check("torchaudio.io", _torchaudio_io)
    check("faster-whisper", _whisper)
    check("f5-tts", _f5)
    check("deep-translator", _translate)
    check("speaker-embeddings", _speaker)
    check("lipsync", _lipsync)
    check("disk", _disk)
    return rep


def cmd_check(_args: argparse.Namespace) -> int:
    rep = environment_report()
    print()
    print(f"  Python {rep['python']}  ({PROJECT_ROOT})")
    print()
    width = max(len(k) for k in rep["checks"])
    for name, st in rep["checks"].items():
        mark = "OK  " if st["ok"] else "MISS"
        print(f"  [{mark}] {name:<{width}}  {st['detail']}")
    print()
    bad = [k for k, v in rep["checks"].items() if not v["ok"]]
    if bad:
        print(f"  {len(bad)} component(s) unavailable: {', '.join(bad)}")
        print("  The pipeline still runs, degrading gracefully where possible.")
    else:
        print("  Everything is available.")
    print()
    return 0


def cmd_languages(_args: argparse.Namespace) -> int:
    print("\n  Supported target languages (Whisper code -> name):\n")
    for code in sorted(LANGUAGES, key=lambda c: LANGUAGES[c]):
        print(f"    {code:<4} {LANGUAGES[code]}")
    print()
    return 0


def cmd_hw(argv: List[str]) -> int:
    """Detect the hardware and show the settings it implies."""
    p = argparse.ArgumentParser(
        prog=f"{PROG} hw",
        description="Detecta CPU, GPU, memória e disco, e mostra os ajustes automáticos.",
    )
    p.add_argument("--refresh", action="store_true", help="ignora o cache e redetecta")
    p.add_argument("--benchmark", action="store_true",
                   help="mede a velocidade real desta máquina (deixa a estimativa precisa)")
    p.add_argument("--json", action="store_true", help="saída em JSON")
    args = p.parse_args(argv)

    from .hardware import detect, recommend, report_lines  # noqa: PLC0415

    hw = detect(deep=True, refresh=args.refresh)
    rec = recommend(hw, measure=args.benchmark)

    if args.json:
        print(json.dumps({"hardware": hw.to_dict(), "recommendation": rec.to_dict()},
                         indent=2, ensure_ascii=False))
        return 0

    print()
    for line in report_lines(hw, rec):
        print(f"  {line}" if line else "")
    print()
    if not args.benchmark:
        print("  Dica: 'dublador hw --benchmark' mede a maquina e deixa a estimativa precisa.")
        print()
    return 0


def cmd_estimate(argv: List[str]) -> int:
    """Work out how long a dub will take, before committing to it."""
    p = argparse.ArgumentParser(
        prog=f"{PROG} estimate",
        description="Estima o tempo total de uma dublagem, etapa por etapa.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
exemplos:
  dublador estimate -d 1h30m
  dublador estimate -d 1h30m --no-lipsync
  dublador estimate -d 5m --speech-ratio 0.6
  dublador estimate -d 1h30m --cores 6 --gflops 400 --vram 8 --gpu-name "RTX 5060"
""",
    )
    p.add_argument("-d", "--duration", required=True,
                   help="duração do vídeo: '1h30m', '90m', '5400' ou '1:30'")
    p.add_argument("--speech-ratio", type=float, default=0.5,
                   help="fração do vídeo que é fala (padrão 0.5; filmes ficam entre 0.4 e 0.6)")
    p.add_argument("--fps", type=float, default=24.0)
    p.add_argument("--nfe-step", type=int, default=0, help="0 = automático")
    p.add_argument("--no-lipsync", action="store_true")
    p.add_argument("--json", action="store_true")
    # "what if" simulation for another machine
    p.add_argument("--cores", type=int, default=0, help="simula N núcleos físicos")
    p.add_argument("--gflops", type=float, default=0.0,
                   help="simula o throughput da CPU (referência: i3-4130 = 108 GFLOPS)")
    p.add_argument("--vram", type=float, default=0.0, help="simula uma GPU com N GB de VRAM")
    p.add_argument("--gpu-name", default="", help="nome da GPU simulada")
    args = p.parse_args(argv)

    from .estimate import estimate_job, parse_duration, simulated_hardware  # noqa: PLC0415
    from .hardware import detect, report_lines  # noqa: PLC0415
    from .utils import human_duration  # noqa: PLC0415

    try:
        seconds = parse_duration(args.duration)
    except ValueError as e:
        print(f"\n  Duração inválida: {e}\n")
        return 2
    if seconds <= 0:
        print("\n  A duração precisa ser maior que zero.\n")
        return 2

    if args.cores or args.gflops or args.vram:
        hw = simulated_hardware(
            physical_cores=args.cores or 4,
            gflops=args.gflops or 108.2,
            vram_gb=args.vram,
            gpu_name=args.gpu_name,
        )
        gflops = args.gflops or 108.2
        simulated = True
    else:
        hw = detect(deep=True)
        gflops = None
        simulated = False

    est = estimate_job(
        seconds, speech_ratio=args.speech_ratio, fps=args.fps, hw=hw,
        nfe_step=args.nfe_step or None, lipsync=not args.no_lipsync, gflops=gflops,
    )

    if args.json:
        print(json.dumps({"estimate": est.to_dict(),
                          "hardware": hw.to_dict()}, indent=2, ensure_ascii=False))
        return 0

    print()
    print("=" * 74)
    print("  ESTIMATIVA DE TEMPO" + ("  (maquina simulada)" if simulated else ""))
    print("=" * 74)
    print(f"  Video            {human_duration(est.video_seconds)}")
    print(f"  Fala estimada    {human_duration(est.speech_seconds)} "
          f"({est.speech_seconds / est.video_seconds:.0%} do video)")
    print(f"  Falas            ~{est.segments}")
    print(f"  Maquina          {hw.summary}  [{est.tiers['tier']}]")
    print(f"  Vai usar         Whisper {est.tiers['whisper']}, "
          f"F5-TTS com {est.tiers['nfe_step']} passos, device {est.device}")
    print()
    print(f"  {'ETAPA':<34}{'TEMPO':>11}   DETALHE")
    print("  " + "-" * 70)
    for s in est.stages:
        print(f"  {s.label:<34}{human_duration(s.seconds):>11}   {s.detail}")
    print("  " + "-" * 70)
    print(f"  {'TOTAL':<34}{human_duration(est.total_seconds):>11}")
    print(f"  {'(faixa realista)':<34}"
          f"{human_duration(est.total_seconds * 0.65) + ' - ' + human_duration(est.total_seconds * 1.70):>11}")
    print()
    if est.alternatives:
        print("  ALTERNATIVAS")
        for name, secs in est.alternatives.items():
            print(f"    {name:<32}{human_duration(secs):>11}")
        print()
    if est.notes:
        print("  ATENCAO")
        for n in est.notes:
            for i, chunk in enumerate(_wrap(n, 68)):
                print(f"    {'-' if i else ' '} {chunk}")
        print()
    return 0


def _wrap(text: str, width: int) -> List[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if cur and len(cur) + 1 + len(w) > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines or [""]


def cmd_gui(argv: List[str]) -> int:
    """Launch the local web interface (no console commands needed)."""
    p = argparse.ArgumentParser(
        prog=f"{PROG} gui",
        description="Abre a interface web local para dublar vídeos sem usar o console.",
    )
    p.add_argument("--port", type=int, default=8760, help="porta (padrão 8760)")
    p.add_argument("--host", default="127.0.0.1", help="endereço (padrão: só local)")
    p.add_argument("--no-browser", action="store_true",
                   help="não abrir o navegador automaticamente")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    try:
        from .gui import serve  # noqa: PLC0415
    except Exception as e:  # noqa: BLE001
        print(f"\n  Não foi possível carregar a interface: {e}\n")
        return 1

    try:
        return serve(args.host, args.port,
                     open_browser=not args.no_browser, verbose=args.verbose)
    except KeyboardInterrupt:
        print("\n  interface encerrada\n")
        return 0
    except OSError as e:
        print(f"\n  Não foi possível abrir a porta {args.port}: {e}")
        print(f"  Tente outra porta:  .\\run.ps1 gui --port 8900\n")
        return 1


# --------------------------------------------------------------------------
# argument parsing
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="Dub a video into another language, cloning each actor's voice with F5-TTS.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  dublador hw                     detecta o hardware e mostra os ajustes ideais
  dublador hw --benchmark         mede a maquina e estima o tempo com precisao
  dublador gui                    abre a interface no navegador
  dublador check
  dublador languages
  dublador movie.mp4 --target pt
  dublador movie.mp4 --target en --source es --whisper-model large-v3
  dublador movie.mp4 --target pt --no-lipsync --background duck
  dublador movie.mp4 --target pt --until translate   (confere antes de sintetizar)

Por padrao o programa detecta CPU, GPU e memoria sozinho e escolhe
dispositivo, threads, modelos e passos de sintese.  Rode 'dublador hw'
para ver o que foi escolhido, ou passe os valores explicitamente para
sobrepor a deteccao.

Presets:
  --fast     small Whisper, 16 denoising steps, no lip sync   (quick preview)
  --quality  large-v3 Whisper, 32 steps, lip sync on          (best result)
""",
    )

    p.add_argument("input", nargs="?", help="input video file (or a name inside ./input)")
    p.add_argument("-t", "--target", "--target-lang", dest="target_lang",
                   help="target language code (e.g. pt, en, es)")
    p.add_argument("-s", "--source", "--source-lang", dest="source_lang",
                   help="source language (default: auto-detect)")
    p.add_argument("-o", "--output", help="output video path")

    g = p.add_argument_group("speech to text")
    g.add_argument("--whisper-model", default="auto",
                   choices=["auto", "tiny", "base", "small", "medium", "large-v3",
                            "distil-large-v3"],
                   help="'auto' escolhe pelo hardware detectado (padrão)")
    g.add_argument("--whisper-compute-type", default="auto",
                   choices=["auto", "int8", "int8_float32", "float32", "float16"],
                   help="'auto' = float16 na GPU, int8 na CPU (padrão)")
    g.add_argument("--beam-size", type=int, default=5)
    g.add_argument("--no-vad", action="store_true", help="disable the VAD filter")

    g = p.add_argument_group("voices")
    g.add_argument("--min-speakers", type=int, default=1)
    g.add_argument("--max-speakers", type=int, default=6)
    g.add_argument("--speaker-threshold", type=float, default=0.0,
                   help="clustering cut; 0 = auto per backend (default). "
                        "Raise it to split voices more aggressively, lower it to merge.")

    g = p.add_argument_group("translation")
    g.add_argument("--translator", default="google",
                   choices=["google", "mymemory", "libre"])

    g = p.add_argument_group("text to speech (F5-TTS)")
    g.add_argument("--f5-model", default="F5TTS_v1_Base",
                   choices=["F5TTS_v1_Base", "F5TTS_v1_Small", "F5TTS_Base",
                            "F5TTS_Small", "E2TTS_Base"])
    g.add_argument("--nfe-step", type=int, default=0,
                   help="passos de denoising: 0 = automático, 16 = rápido, 32 = melhor")
    g.add_argument("--cfg-strength", type=float, default=2.0)
    g.add_argument("--tts-speed", type=float, default=1.0)
    g.add_argument("--no-fix-duration", action="store_true",
                   help="let F5-TTS choose its own length instead of filling the slot")

    g = p.add_argument_group("timing")
    g.add_argument("--max-stretch", type=float, default=1.40)
    g.add_argument("--min-stretch", type=float, default=0.72)
    g.add_argument("--segment-gap", type=float, default=0.06)

    g = p.add_argument_group("video / audio out")
    g.add_argument("--no-lipsync", action="store_true", help="skip Wav2Lip")
    g.add_argument("--lipsync-force", action="store_true",
                   help="run Wav2Lip even on long videos (very slow on CPU)")
    g.add_argument("--lipsync-max-seconds", type=float, default=90.0,
                   help="skip lip sync above this duration (default 90)")
    g.add_argument("--lipsync-resize", type=int, default=1, choices=[1, 2, 4],
                   help="downscale before Wav2Lip: 2 is ~4x faster, softer result")
    g.add_argument("--lipsync-batch", type=int, default=8)
    g.add_argument("--background", default="none", choices=["none", "duck", "separate"],
                   help="what to do with the original soundtrack")
    g.add_argument("--voice-gain", type=float, default=1.5, help="dB")
    g.add_argument("--background-gain", type=float, default=-11.0, help="dB")

    g = p.add_argument_group("runtime")
    g.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    g.add_argument("--threads", type=int, default=0)
    g.add_argument("--seed", type=int, default=1234)
    g.add_argument("--job-name", default="job")
    g.add_argument("--no-resume", action="store_true",
                   help="ignore cached stages and recompute everything")
    g.add_argument("--until", default="all",
                   choices=["all", "extract", "transcribe", "voices", "translate",
                            "tts", "assemble", "finish"],
                   help="stop after this stage; useful to inspect the translation "
                        "before committing to a long synthesis run")
    g.add_argument("--transcribe-only", action="store_true",
                   help="only transcribe (cheap smoke test)")
    g.add_argument("-v", "--verbose", action="store_true")

    g = p.add_argument_group("presets")
    g.add_argument("--fast", action="store_true",
                   help="small Whisper + 16 steps + no lip sync")
    g.add_argument("--quality", action="store_true",
                   help="large-v3 Whisper + 32 steps + lip sync")

    return p


def cfg_from_args(args: argparse.Namespace) -> DubladorConfig:
    cfg = DubladorConfig(
        input_video=args.input or "",
        output_video=args.output or "",
        target_lang=args.target_lang or "pt",
        source_lang=args.source_lang,
        whisper_model=args.whisper_model,
        whisper_compute_type=args.whisper_compute_type,
        whisper_beam_size=args.beam_size,
        vad_filter=not args.no_vad,
        translator=args.translator,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
        speaker_threshold=args.speaker_threshold,
        f5_model=args.f5_model,
        tts_nfe_step=args.nfe_step,
        tts_cfg_strength=args.cfg_strength,
        tts_speed=args.tts_speed,
        tts_fix_duration=not args.no_fix_duration,
        max_stretch=args.max_stretch,
        min_stretch=args.min_stretch,
        segment_gap=args.segment_gap,
        lipsync=not args.no_lipsync,
        lipsync_force=args.lipsync_force,
        lipsync_max_seconds=args.lipsync_max_seconds,
        lipsync_resize_factor=args.lipsync_resize,
        lipsync_batch=args.lipsync_batch,
        background_mode=args.background,
        voice_gain_db=args.voice_gain,
        background_gain_db=args.background_gain,
        device=args.device,
        threads=args.threads,
        seed=args.seed,
        job_name=args.job_name,
        resume=not args.no_resume,
    )

    if args.fast:
        cfg.whisper_model = "small"
        cfg.tts_nfe_step = 16
        cfg.lipsync = False
    if args.quality:
        cfg.whisper_model = "large-v3"
        cfg.tts_nfe_step = 32
        cfg.lipsync = True
    return cfg


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    # sub-commands that need no video
    if argv and argv[0] in ("check", "doctor"):
        setup_logging(False)
        return cmd_check(argparse.Namespace())
    if argv and argv[0] in ("languages", "langs"):
        return cmd_languages(argparse.Namespace())
    if argv and argv[0] in ("hw", "hardware", "specs"):
        return cmd_hw(argv[1:])
    if argv and argv[0] in ("estimate", "estimar", "tempo"):
        return cmd_estimate(argv[1:])
    if argv and argv[0] in ("gui", "ui", "interface"):
        return cmd_gui(argv[1:])
    if argv and argv[0] in ("--version", "version"):
        from .pipeline import __version__  # noqa: PLC0415

        print(f"{PROG} {__version__}")
        return 0

    parser = build_parser()
    args = parser.parse_args(argv)

    ensure_dirs()
    log_file = PROJECT_ROOT / "logs" / "dublador.log"
    setup_logging(args.verbose, log_file)

    if not args.input:
        parser.print_help()
        print("\n  error: an input video is required (or use 'check' / 'languages')")
        return 2
    if not args.target_lang:
        print("\n  error: --target is required, e.g. --target pt")
        print("  run 'dublador languages' to list the codes\n")
        return 2

    try:
        normalize_lang(args.target_lang)
    except ValueError as e:
        print(f"\n  error: {e}\n")
        return 2

    cfg = cfg_from_args(args)

    from .pipeline import run_pipeline, run_transcribe_only  # noqa: PLC0415

    try:
        if args.transcribe_only:
            res = run_transcribe_only(cfg)
            print()
            print(json.dumps(res, indent=2, ensure_ascii=False))
            return 0

        result = run_pipeline(cfg, until=args.until)
        print()
        if result.output_video is None:
            print(f"  stopped after stage: {result.stopped_after}")
            print(f"  work dir           : {result.work_dir}")
            print(f"  elapsed            : {human_duration(result.elapsed)}")
            print()
            print("  Re-run the same command without --until to continue.")
            print()
            return 0
        print(f"  dubbed video: {result.output_video}")
        print(f"  elapsed     : {human_duration(result.elapsed)}")
        print()
        return 0

    except KeyboardInterrupt:
        LOG.warning("")
        LOG.warning("interrupted by the user. Progress is checkpointed - re-run the "
                    "same command to resume from the last completed stage.")
        return 130
    except Exception as e:  # noqa: BLE001
        LOG.error("")
        LOG.error("FAILED: %s", e)
        if args.verbose:
            import traceback  # noqa: PLC0415

            traceback.print_exc()
        else:
            LOG.error("re-run with -v for the full traceback")
        return 1


if __name__ == "__main__":
    sys.exit(main())
