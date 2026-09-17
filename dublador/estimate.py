"""Per-stage time estimation for a whole job.

Dubbing is not one cost, it is seven with completely different scaling laws:

* transcription and synthesis scale with **how much speech there is** and love
  a GPU;
* translation scales with the **number of lines** and is bound by a web API,
  not by hardware at all;
* **lip sync scales with video length** (it is per frame, and re-animates the
  whole timeline), which is why it dominates on long material;
* encoding is the only stage that cares about resolution.

Every constant below is documented and, where possible, anchored on a real
measurement from the reference machine (Intel i3-4130, 2 cores).  The output is
deliberately a *range*: these are engineering estimates, not promises.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .hardware import HardwareInfo, _gpu_boost, detect, estimate_speed
from .utils import human_duration

# --------------------------------------------------------------------------
# measured / documented constants
# --------------------------------------------------------------------------
#: Whisper throughput on the REFERENCE CPU (i3-4130, 2 physical cores, int8),
#: in "seconds of audio per second of wall clock".  Measured ~0.7 for `small`.
#: Bigger models are proportionally slower.
WHISPER_CPU_RATE = {
    "tiny": 3.0, "base": 1.8, "small": 0.7, "medium": 0.35,
    "large-v3": 0.2, "distil-large-v3": 0.6,
}
#: Same models on a CUDA GPU of the "entry" class (6-8 GB, e.g. RTX 5060).
WHISPER_GPU_RATE = {
    "tiny": 90.0, "base": 60.0, "small": 45.0, "medium": 30.0,
    "large-v3": 18.0, "distil-large-v3": 45.0,
}
#: Speaker embedding extraction: measured 3.6x realtime on the reference CPU
#: (3 clips, 6.2 s of speech in 1.7 s), but thousands of short clips carry
#: per-segment overhead, so the model floors at a fixed cost per line.
DIARIZE_CPU_RATE = 2.0
DIARIZE_PER_SEGMENT_S = 0.20

#: One translation round trip, and how many run concurrently.
TRANSLATE_LATENCY_S = 0.35
TRANSLATE_WORKERS = 8

#: Extra cost per synthesised line beyond the audio itself (prompt encoding,
#: reference pre-processing, file I/O).
TTS_PER_SEGMENT_S = 0.9

#: Face detection dominates Wav2Lip and runs on EVERY frame.
LIPSYNC_SEC_PER_FRAME = {"cpu": 1.00, "entry": 0.055, "mid": 0.035, "high": 0.022}

#: Video re-encode after lip sync, as "seconds of encode per second of video"
#: (i.e. 1.0 = realtime).  1080p.  NVENC is far faster than x264 on 6 cores.
ENCODE_RATE = {"nvenc": 0.12, "cpu": 1.10}
#: Demuxing/decoding audio for the whole film.
EXTRACT_RATE = 0.02
#: Stream-copy muxing (no re-encode).
MUX_SECONDS = 20.0

#: Uncertainty band applied to the final figure.
LOW_FACTOR, HIGH_FACTOR = 0.65, 1.70


@dataclass
class StageEstimate:
    key: str
    label: str
    seconds: float
    detail: str
    bound_by: str = "hardware"      # hardware | network | io

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["human"] = human_duration(self.seconds)
        return d


@dataclass
class JobEstimate:
    video_seconds: float
    speech_seconds: float
    segments: int
    fps: float
    device: str
    tiers: Dict[str, str] = field(default_factory=dict)
    stages: List[StageEstimate] = field(default_factory=list)
    total_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)
    alternatives: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "video_seconds": self.video_seconds,
            "video_human": human_duration(self.video_seconds),
            "speech_seconds": self.speech_seconds,
            "speech_human": human_duration(self.speech_seconds),
            "segments": self.segments,
            "fps": self.fps,
            "device": self.device,
            "total_seconds": self.total_seconds,
            "total_human": human_duration(self.total_seconds),
            "range_human": [human_duration(self.total_seconds * LOW_FACTOR),
                            human_duration(self.total_seconds * HIGH_FACTOR)],
            "stages": [s.to_dict() for s in self.stages],
            "notes": self.notes,
            "alternatives": {k: human_duration(v) for k, v in self.alternatives.items()},
        }


def _gpu_class(hw: HardwareInfo) -> str:
    gpu = hw.best_gpu
    if not gpu:
        return "cpu"
    if gpu.vram_gb >= 16:
        return "high"
    if gpu.vram_gb >= 10:
        return "mid"
    return "entry"


def estimate_job(
    video_seconds: float,
    *,
    speech_ratio: float = 0.5,
    fps: float = 24.0,
    hw: Optional[HardwareInfo] = None,
    nfe_step: Optional[int] = None,
    lipsync: bool = True,
    lipsync_max_seconds: float = 90.0,
    whisper_model: Optional[str] = None,
    resolution: str = "1080p",
    gflops: Optional[float] = None,
) -> JobEstimate:
    """Estimate wall-clock time for a full dub.

    ``speech_ratio`` is the fraction of the runtime that is actually dialogue
    -- typically 0.4-0.6 for a feature film.  It matters enormously: synthesis
    cost tracks *speech*, while lip-sync cost tracks the *whole video*.
    """
    hw = hw or detect(deep=True)
    dev = hw.device
    on_gpu = bool(hw.best_gpu and hw.best_gpu.available)
    gclass = _gpu_class(hw)

    # Use the very same self-tuning the pipeline uses, so the estimate always
    # describes what the program would really do on this machine.
    from .hardware import recommend  # noqa: PLC0415

    rec = recommend(hw, nfe_step=nfe_step)
    nfe = nfe_step or rec.nfe_step
    model = whisper_model or rec.whisper_model

    speech = max(1.0, video_seconds * max(0.05, min(1.0, speech_ratio)))
    segments = max(1, int(speech / 2.5))            # ~2.5 s per line
    notes: List[str] = []

    speed = estimate_speed(hw, nfe_step=nfe, gflops=gflops)
    #: Raw CPU throughput vs the reference box.  Transcription and diarisation
    #: scale with this; only F5-TTS cares about the step count.
    cpu_mult = max(0.5, float(speed.get("cpu_vs_reference") or 1.0))

    # ---- 1. audio extraction --------------------------------------------
    extract = video_seconds * EXTRACT_RATE

    # ---- 2. transcription ------------------------------------------------
    if on_gpu:
        base = WHISPER_GPU_RATE.get(model, 20.0)
        rate = base * max(0.4, _gpu_boost(hw.best_gpu) / 20.0)
    else:
        # WHISPER_CPU_RATE is already expressed for the reference machine, so
        # the machine factor multiplies it directly (multiplier is 1.0 there).
        base = WHISPER_CPU_RATE.get(model, 0.7)
        rate = base * cpu_mult
    rate = max(0.05, rate)
    transcribe = video_seconds / rate

    # ---- 3. voice analysis ----------------------------------------------
    diar_rate = DIARIZE_CPU_RATE * cpu_mult * (6.0 if on_gpu else 1.0)
    diarize = max(speech / max(0.2, diar_rate), segments * DIARIZE_PER_SEGMENT_S)

    # ---- 4. translation (network bound) ---------------------------------
    translate = segments * TRANSLATE_LATENCY_S / TRANSLATE_WORKERS
    translate = max(translate, 20.0)     # cold start + cache writes

    # ---- 5. synthesis ----------------------------------------------------
    tts = speech * speed["realtime_factor"] + segments * TTS_PER_SEGMENT_S

    # ---- 6. lip sync -----------------------------------------------------
    #: How long lip sync would take if it were actually run.
    forced_lipsync_s = video_seconds * fps * LIPSYNC_SEC_PER_FRAME[
        gclass if on_gpu else "cpu"
    ]
    # ...and what the pipeline actually does by default, which for anything
    # longer than lipsync_max_seconds is to skip it entirely.
    auto_skipped = lipsync and video_seconds > lipsync_max_seconds
    run_lipsync = lipsync and not auto_skipped
    lipsync_s = forced_lipsync_s if run_lipsync else 0.0
    lipsync_detail = f"{int(video_seconds * fps):,}".replace(",", ".") + " quadros"
    if not on_gpu and run_lipsync:
        lipsync_detail += (f" a {LIPSYNC_SEC_PER_FRAME['cpu']:.2f} s/quadro "
                           f"(CPU, sem GPU)")
    if auto_skipped:
        lipsync_detail = "pulado automaticamente (vídeo longo)"

    # ---- 7. encode / mux -------------------------------------------------
    erate = ENCODE_RATE["nvenc"] if on_gpu else ENCODE_RATE["cpu"]
    full_encode = video_seconds * erate
    encode = full_encode if run_lipsync else MUX_SECONDS
    if run_lipsync and not on_gpu:
        notes.append("A re-codificacao em CPU (x264) e um custo grande; "
                     "uma GPU com NVENC reduziria bastante.")

    # ---- assembly (numpy timeline + wav writes) --------------------------
    assemble = segments * 0.01 + video_seconds * 0.002

    stages = [
        StageEstimate("extract", "Extração de áudio", extract,
                      f"{video_seconds / 60:.0f} min de vídeo", "io"),
        StageEstimate("transcribe", "Transcrição (Whisper)", transcribe,
                      f"{model} em {'GPU' if on_gpu else 'CPU'}, ~{rate:.1f}x tempo real"),
        StageEstimate("diarize", "Análise de vozes", diarize,
                      f"{segments} falas para agrupar"),
        StageEstimate("translate", "Tradução", translate,
                      f"{segments} segmentos via API web", "network"),
        StageEstimate("tts", "Síntese de voz (F5-TTS)", tts,
                      f"{human_duration(speech)} de fala, {nfe} passos, "
                      f"{speed['realtime_factor']:.2f}x tempo real"),
        StageEstimate("lipsync", "Sincronização labial (Wav2Lip)", lipsync_s,
                      lipsync_detail),
        StageEstimate("assemble", "Montagem da linha do tempo", assemble,
                      f"{segments} clipes"),
        StageEstimate("encode", "Codificação final / mux", encode,
                      "re-codificação do vídeo" if run_lipsync else "cópia direta"),
    ]

    total = sum(s.seconds for s in stages)

    # ---- alternatives ----------------------------------------------------
    tts_16 = (speech * estimate_speed(hw, nfe_step=16, gflops=gflops)["realtime_factor"]
              + segments * TTS_PER_SEGMENT_S)
    base_no_lip = total - lipsync_s - encode + MUX_SECONDS
    alt: Dict[str, float] = {}
    if auto_skipped:
        alt["com lip sync forçado (--lipsync-force)"] = (
            base_no_lip - MUX_SECONDS + forced_lipsync_s + full_encode
        )
    else:
        alt["sem lip sync"] = base_no_lip
    if nfe != 16:
        alt["com 16 passos em vez de " + str(nfe)] = base_no_lip - tts + tts_16

    notes.append(
        "A traducao e limitada pela API, nao pelo hardware: o tier gratuito do "
        "MyMemory nao cobre um filme inteiro, e o Google nao oficial sofre "
        "rate-limit. Para 1h30 conte com um servico pago (DeepL) ou um modelo local."
    )
    if auto_skipped:
        notes.append(
            f"O lip sync NAO esta no total acima: em videos com mais de "
            f"{lipsync_max_seconds:.0f} s o pipeline o pula sozinho. Forcar "
            f"custaria cerca de {human_duration(forced_lipsync_s)} a mais."
        )
    if run_lipsync and video_seconds > 600:
        notes.append(
            "O lip sync em filme inteiro raramente compensa: com cortes, "
            "movimento de camera e varios rostos o resultado degrada, e e a "
            "etapa mais cara."
        )

    return JobEstimate(
        video_seconds=video_seconds, speech_seconds=speech, segments=segments,
        fps=fps, device=dev,
        tiers={"tier": hw.tier, "gpu_class": gclass,
               "whisper": model, "nfe_step": str(nfe)},
        stages=stages, total_seconds=total, notes=notes, alternatives=alt,
    )


def simulated_hardware(*, physical_cores: int, gflops: float, vram_gb: float = 0.0,
                       gpu_name: str = "", device: str = "") -> HardwareInfo:
    """Build a synthetic :class:`HardwareInfo` to answer 'what if' questions."""
    from .hardware import CPUInfo, GPUInfo, MemoryInfo

    gpus: List[GPUInfo] = []
    if vram_gb > 0:
        gpus.append(GPUInfo(vendor="nvidia", name=gpu_name or f"GPU {vram_gb:.0f} GB",
                            vram_gb=vram_gb, backend=device or "cuda", available=True))
    hw = HardwareInfo(
        cpu=CPUInfo(model=f"simulado ({physical_cores} nucleos)",
                    physical_cores=physical_cores, logical_cores=physical_cores * 2,
                    arch="AMD64", ct2_compute_types=["int8", "float32"]),
        memory=MemoryInfo(total_gb=24.0, available_gb=20.0),
        gpus=gpus, disk_free_gb=200.0, disk_total_gb=1000.0,
        os_name="simulado", python="3.11", torch="(simulado)", cuda="",
    )
    from .hardware import classify

    hw.tier = classify(hw)
    return hw


def parse_duration(text: str) -> float:
    """Accept '5400', '90m', '1h30m', '1:30' -> seconds."""
    t = str(text).strip().lower().replace(" ", "")
    if not t:
        raise ValueError("empty duration")
    if ":" in t:
        parts = [float(p) for p in t.split(":")]
        total = 0.0
        for p in parts:
            total = total * 60 + p
        return total
    total = 0.0
    num = ""
    for ch in t:
        if ch.isdigit() or ch == ".":
            num += ch
            continue
        if ch in ("h",):
            total += float(num or 0) * 3600
        elif ch in ("m",):
            total += float(num or 0) * 60
        elif ch in ("s",):
            total += float(num or 0)
        else:
            raise ValueError(f"cannot parse duration {text!r}")
        num = ""
    if num:
        total += float(num)
    return total
