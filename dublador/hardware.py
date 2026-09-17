"""Automatic hardware detection and self-tuning.

The pipeline should not need to be told what machine it is running on.  This
module probes CPU, memory, GPU and disk, classifies the host into a capability
tier, and derives concrete settings (device, compute type, thread counts, model
sizes, denoising steps, whether lip sync is realistic) from what it finds.

Design constraints
------------------
* **No hard dependencies.** ``psutil`` and ``torch`` are used when present, but
  every probe has a dependency-free fallback (``ctypes`` for memory on Windows,
  the registry/sysfs for GPU presence, ``os`` for cores).
* **Cheap by default.** Importing ``torch`` costs seconds, so the expensive GPU
  probe is cached on disk and can be skipped with ``deep=False``.
* **Honest estimates.** :func:`estimate_speed` extrapolates from a real
  measurement taken on this project's reference machine (Intel i3-4130, where
  F5-TTS v1 Base at 16 steps ran at ~55x realtime) scaled by a measured,
  machine-independent numpy/OpenBLAS throughput benchmark.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import MODELS_DIR, PROJECT_ROOT
from .utils import LOG, read_json, write_json

CACHE_FILE = PROJECT_ROOT / ".cache" / "hardware.json"
CACHE_TTL = 6 * 3600  # re-probe every 6 hours
#: Bump when the shape of the cached record changes.
CACHE_VERSION = 2


# --------------------------------------------------------------------------
# data model
# --------------------------------------------------------------------------
@dataclass
class CPUInfo:
    model: str = "unknown"
    physical_cores: int = 0
    logical_cores: int = 0
    max_mhz: float = 0.0
    arch: str = ""
    #: compute types CTranslate2 (faster-whisper's backend) actually supports,
    #: which is the practical answer to "can this CPU do fast int8 inference?"
    ct2_compute_types: List[str] = field(default_factory=list)

    @property
    def has_avx2(self) -> bool:
        return any("int8" in c for c in self.ct2_compute_types) or "AVX2" in self.model.upper()


@dataclass
class MemoryInfo:
    total_gb: float = 0.0
    available_gb: float = 0.0

    @property
    def pressure(self) -> str:
        if not self.total_gb:
            return "unknown"
        frac = self.available_gb / self.total_gb
        if frac < 0.15:
            return "critical"
        if frac < 0.30:
            return "high"
        return "ok"


@dataclass
class GPUInfo:
    vendor: str = "none"          # nvidia | amd | intel | apple | none
    name: str = ""
    vram_gb: float = 0.0
    backend: str = "cpu"          # cuda | rocm | mps | xpu | cpu
    available: bool = False
    driver: str = ""

    @property
    def label(self) -> str:
        if not self.available:
            return "nenhuma GPU utilizável"
        return f"{self.name} ({self.vram_gb:.1f} GB VRAM, {self.backend})"


@dataclass
class HardwareInfo:
    cpu: CPUInfo = field(default_factory=CPUInfo)
    memory: MemoryInfo = field(default_factory=MemoryInfo)
    gpus: List[GPUInfo] = field(default_factory=list)
    disk_free_gb: float = 0.0
    disk_total_gb: float = 0.0
    os_name: str = ""
    python: str = ""
    torch: str = ""
    cuda: str = ""
    tier: str = "unknown"
    notes: List[str] = field(default_factory=list)
    probed_at: float = field(default_factory=time.time)

    # -- convenience -------------------------------------------------------
    @property
    def best_gpu(self) -> Optional[GPUInfo]:
        usable = [g for g in self.gpus if g.available]
        return max(usable, key=lambda g: g.vram_gb) if usable else None

    @property
    def device(self) -> str:
        g = self.best_gpu
        return g.backend if g else "cpu"

    @property
    def summary(self) -> str:
        g = self.best_gpu
        return (f"{self.cpu.physical_cores}C/{self.cpu.logical_cores}T CPU, "
                f"{self.memory.total_gb:.1f} GB RAM"
                + (f", {g.name} {g.vram_gb:.1f} GB" if g else ", sem GPU"))

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["device"] = self.device
        d["summary"] = self.summary
        d["cache_version"] = CACHE_VERSION
        return d


# --------------------------------------------------------------------------
# probes
# --------------------------------------------------------------------------
def _cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg  # noqa: PLC0415

            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return str(winreg.QueryValueEx(k, "ProcessorNameString")[0]).strip()
        except Exception:  # noqa: BLE001
            pass
    name = platform.processor() or ""
    if not name:
        try:
            with open("/proc/cpuinfo", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    return name or "unknown"


def _physical_cores() -> Tuple[int, int]:
    logical = os.cpu_count() or 1
    # psutil is the only portable way to get physical cores.
    try:
        import psutil  # noqa: PLC0415

        phys = psutil.cpu_count(logical=False) or logical
        return int(phys), logical
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "win32":
        # Derive physical cores from the ProcessorNameString + env fallbacks.
        n = os.environ.get("NUMBER_OF_PROCESSORS")
        logical_env = int(n) if n and n.isdigit() else logical
        # Without hyper-threading information assume 2 threads per core, which
        # is right for essentially every x86 chip since 2008.
        return max(1, logical_env // 2), logical_env
    return logical, logical


def _ct2_compute_types() -> List[str]:
    """What compute types CTranslate2 supports here (int8 => AVX2 present)."""
    try:
        import ctranslate2  # noqa: PLC0415

        types = ctranslate2.get_supported_compute_types("cpu")
        return sorted(str(t) for t in types)
    except Exception:  # noqa: BLE001
        return []


def _memory() -> MemoryInfo:
    try:
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        return MemoryInfo(round(vm.total / 1024 ** 3, 2), round(vm.available / 1024 ** 3, 2))
    except Exception:  # noqa: BLE001
        pass
    if sys.platform == "win32":
        try:
            import ctypes  # noqa: PLC0415

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = MEMORYSTATUSEX()
            st.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st)):
                return MemoryInfo(round(st.ullTotalPhys / 1024 ** 3, 2),
                                  round(st.ullAvailPhys / 1024 ** 3, 2))
        except Exception:  # noqa: BLE001
            pass
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            info = {}
            for line in fh:
                k, _, v = line.partition(":")
                info[k.strip()] = float(v.strip().split()[0]) / 1024 ** 2
            total = info.get("MemTotal", 0.0)
            avail = info.get("MemAvailable", info.get("MemFree", 0.0))
            return MemoryInfo(round(total, 2), round(avail, 2))
    except OSError:
        pass
    return MemoryInfo()


def _gpus(deep: bool = True) -> Tuple[List[GPUInfo], str, str, List[str]]:
    """Return (gpus, torch_version, cuda_version, notes)."""
    gpus: List[GPUInfo] = []
    notes: List[str] = []
    torch_version = ""
    cuda_version = ""

    if not deep:
        return gpus, torch_version, cuda_version, notes

    try:
        import torch  # noqa: PLC0415

        torch_version = torch.__version__
        cuda_version = getattr(getattr(torch, "version", None), "cuda", "") or ""

        # ---- NVIDIA / ROCm -------------------------------------------------
        try:
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    props = torch.cuda.get_device_properties(i)
                    name = props.name
                    is_rocm = "rocm" in (torch_version or "").lower() or "amd" in name.lower()
                    gpus.append(GPUInfo(
                        vendor="amd" if is_rocm else "nvidia",
                        name=name,
                        vram_gb=round(props.total_memory / 1024 ** 3, 2),
                        backend="rocm" if is_rocm else "cuda",
                        available=True,
                        driver=f"compute {props.major}.{props.minor}",
                    ))
            else:
                if getattr(torch.version, "cuda", None):
                    notes.append(
                        "PyTorch was built with CUDA but no CUDA device is visible "
                        "(driver missing, or the GPU is not supported)."
                    )
        except Exception as e:  # noqa: BLE001
            notes.append(f"CUDA probe failed: {e}")

        # ---- Intel XPU -----------------------------------------------------
        try:
            xpu = getattr(torch, "xpu", None)
            if xpu is not None and xpu.is_available():
                for i in range(xpu.device_count()):
                    props = xpu.get_device_properties(i)
                    gpus.append(GPUInfo(
                        vendor="intel", name=getattr(props, "name", "Intel XPU"),
                        vram_gb=round(getattr(props, "total_memory", 0) / 1024 ** 3, 2),
                        backend="xpu", available=True,
                    ))
        except Exception:  # noqa: BLE001
            pass

        # ---- Apple MPS -----------------------------------------------------
        try:
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                gpus.append(GPUInfo(vendor="apple", name="Apple Silicon (MPS)",
                                    vram_gb=0.0, backend="mps", available=True,
                                    driver="unified memory"))
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        notes.append(f"PyTorch not importable for GPU probing ({e}).")

    if not gpus:
        integ = _integrated_gpu_name()
        if integ:
            gpus.append(GPUInfo(vendor=_vendor_of(integ), name=integ, backend="cpu",
                                available=False,
                                driver="integrada - sem suporte a CUDA"))
    return gpus, torch_version, cuda_version, notes


def _vendor_of(name: str) -> str:
    n = name.lower()
    if "nvidia" in n or "geforce" in n or "quadro" in n or "rtx" in n or "gtx" in n:
        return "nvidia"
    if "amd" in n or "radeon" in n:
        return "amd"
    if "intel" in n:
        return "intel"
    return "unknown"


def _integrated_gpu_name() -> str:
    """Find the display adapter without importing torch (registry / sysfs)."""
    if sys.platform == "win32":
        try:
            import winreg  # noqa: PLC0415

            key = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, key) as k:
                i = 0
                while True:
                    try:
                        sub = winreg.EnumKey(k, i)
                    except OSError:
                        break
                    i += 1
                    try:
                        with winreg.OpenKey(k, sub) as sk:
                            desc = winreg.QueryValueEx(sk, "DriverDesc")[0]
                        if desc:
                            return str(desc)
                    except OSError:
                        continue
        except Exception:  # noqa: BLE001
            pass
    else:
        try:
            base = Path("/sys/class/drm")
            for card in sorted(base.glob("card*/device/vendor")):
                vendor = card.read_text(encoding="utf-8").strip()
                name_file = card.parent / "device"
                label = ""
                try:
                    label = (name_file / "uevent").read_text(encoding="utf-8")
                    label = next((l.split("=", 1)[1] for l in label.splitlines()
                                  if l.startswith("DRIVER=")), "")
                except OSError:
                    pass
                names = {"0x10de": "NVIDIA", "0x1002": "AMD", "0x8086": "Intel"}
                return f"{names.get(vendor, 'GPU')} {label}".strip()
        except Exception:  # noqa: BLE001
            pass
    return ""


# --------------------------------------------------------------------------
# benchmark / speed model
# --------------------------------------------------------------------------
def benchmark(seconds: float = 1.5, samples: int = 5) -> Dict[str, float]:
    """Measure this machine's practical vector-math throughput.

    Uses numpy (which the pipeline already requires) so it reflects OpenBLAS,
    i.e. roughly the same silicon the F5-TTS and Whisper kernels use.

    Reports the **best** of several short samples rather than the mean: BLAS
    peak is far more stable than the average, which is polluted by scheduler
    noise and thermal ramp-up, and an unstable calibration would silently skew
    every ETA the tool prints.
    """
    out: Dict[str, float] = {"gflops": 0.0, "seconds": 0.0, "samples": float(samples)}
    try:
        import numpy as np  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        return out

    n = 512
    a = np.random.rand(n, n).astype(np.float32)
    b = np.random.rand(n, n).astype(np.float32)
    for _ in range(3):  # warm up
        a @ b

    flops_per_mul = 2.0 * n ** 3
    per = max(0.2, seconds / max(1, samples))
    best = 0.0
    total = 0.0
    for _ in range(max(1, samples)):
        iters = 0
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < per:
            a @ b
            iters += 1
        elapsed = time.perf_counter() - t0
        total += elapsed
        if elapsed > 0:
            best = max(best, iters * flops_per_mul / elapsed / 1e9)

    out["seconds"] = round(total, 3)
    out["gflops"] = round(best, 1)
    return out


#: numpy/OpenBLAS float32 matmul throughput **measured on the reference machine**
#: (Intel i3-4130, 2 physical cores) using the same best-of-5 method as
#: :func:`benchmark`: 108.2 GFLOPS.  Every timing quoted in the README was taken
#: on that box, so it is the natural zero point for extrapolating elsewhere.
REFERENCE_GFLOPS = 108.2
#: On that same machine, F5-TTS v1 Base at nfe_step=16 ran at ~55x realtime
#: (7.0 s of speech generated in 6m27s during the end-to-end verification).
REFERENCE_REALTIME_FACTOR = 55.0

CALIBRATION_FILE = PROJECT_ROOT / ".cache" / "speed_calibration.json"


def _gpu_boost(gpu: GPUInfo) -> float:
    """How much faster this GPU is than *this machine's CPU* on the F5-TTS DiT.

    We cannot benchmark a GPU we may not have, so this is a documented rule of
    thumb rather than a measurement.  It is anchored on published F5-TTS
    throughput: a current 60-class card sits near a 0.25 realtime factor where
    an entry-level 6-core CPU sits near 5.2, implying a speed-up around 20x.
    VRAM is the best available proxy for the card's class.  The relative rank
    matters far more than the absolute value, and the result is exposed to the
    user as a range, never as a promise.
    """
    if gpu.backend == "mps":
        return 6.0          # Apple unified memory, no tensor-core path for this model
    if gpu.backend == "xpu":
        return 8.0
    vram = gpu.vram_gb
    if vram >= 16:          # 4080 / 4090 / 5090 class
        return 45.0
    if vram >= 10:          # 4070 / 3090 class
        return 30.0
    if vram >= 6:           # 4060 / 5060 / 3060 class
        return 20.0
    if vram >= 4:           # 3050 / 1650 class
        return 12.0
    return 6.0


def estimate_speed(hw: Optional[HardwareInfo] = None, *, nfe_step: int = 16,
                   measure: bool = False,
                   gflops: Optional[float] = None) -> Dict[str, Any]:
    """Estimate how much slower than realtime synthesis will be.

    ``realtime_factor`` of 55 means "10 s of speech takes 550 s".  A value
    **below 1.0 means faster than realtime** -- which is the normal case on a
    discrete GPU, so the result must not be clamped at 1.0.

    Pass ``gflops`` to describe a machine other than this one (used by
    ``dublador estimate`` for "what if" questions).
    """
    hw = hw or detect(deep=True)

    if gflops is None:
        cached = read_json(CALIBRATION_FILE, default=None)
        if isinstance(cached, dict) and cached.get("gflops"):
            gflops = float(cached["gflops"])
        elif measure:
            bench = benchmark()
            gflops = bench["gflops"]
            if gflops:
                write_json(CALIBRATION_FILE, {"gflops": gflops, "at": time.time()})
        else:
            gflops = 0.0

    scale = (gflops / REFERENCE_GFLOPS) if gflops else 1.0
    # Clamp: a microbenchmark is a proxy, not a measurement of the real kernels.
    scale = max(0.25, min(8.0, scale))
    cores_ratio = max(0.5, hw.cpu.physical_cores / 2.0)
    #: Raw CPU throughput relative to the reference box, independent of the
    #: synthesis step count.  Other stages (transcription, diarisation) scale
    #: with *this* number, not with the F5-TTS realtime factor.
    cpu_vs_reference = max(0.5, scale * cores_ratio)
    cpu_factor = REFERENCE_REALTIME_FACTOR * (nfe_step / 16.0) / cpu_vs_reference

    gpu = hw.best_gpu
    if gpu and gpu.available:
        boost = _gpu_boost(gpu)
        # Floor at 0.03 (~33x faster than realtime): no consumer card does
        # better than that on a 336M-parameter DiT, so anything lower means the
        # heuristic has gone wrong rather than "the GPU is amazing".
        estimate = max(0.03, cpu_factor / boost)
    else:
        estimate = cpu_factor

    return {
        "realtime_factor": round(estimate, 3),
        "cpu_realtime_factor": round(cpu_factor, 1),
        "cpu_vs_reference": round(cpu_vs_reference, 2),
        "gpu_boost": round(_gpu_boost(gpu), 1) if (gpu and gpu.available) else 0.0,
        "gflops": gflops,
        "calibrated": bool(gflops),
        "method": "measured numpy throughput" if gflops else "core count only",
    }


# --------------------------------------------------------------------------
# classification + recommendations
# --------------------------------------------------------------------------
def classify(hw: HardwareInfo) -> str:
    gpu = hw.best_gpu
    if gpu and gpu.available:
        if gpu.backend == "mps":
            return "gpu-apple"
        if gpu.vram_gb >= 10:
            return "gpu-strong"
        if gpu.vram_gb >= 6:
            return "gpu-modest"
        return "gpu-small"

    cores = hw.cpu.physical_cores
    ram = hw.memory.total_gb
    if cores >= 8 and ram >= 16:
        return "cpu-strong"
    # Two real cores with enough RAM can still finish a dub -- this project was
    # verified end-to-end on an i3-4130 (2 cores, 8 GB) -- so that is "modest",
    # not "minimal".  Reserve "minimal" for genuinely unable hardware.
    if cores >= 2 and ram >= 6:
        return "cpu-modest"
    return "cpu-minimal"


TIER_LABEL = {
    "gpu-strong": "GPU forte",
    "gpu-modest": "GPU média",
    "gpu-small": "GPU pequena",
    "gpu-apple": "Apple Silicon",
    "cpu-strong": "CPU forte",
    "cpu-modest": "CPU modesta",
    "cpu-minimal": "CPU fraca (sem GPU)",
}


@dataclass
class Recommendation:
    tier: str
    tier_label: str
    device: str
    whisper_model: str
    whisper_compute_type: str
    threads: int
    nfe_step: int
    f5_model: str
    lipsync: bool
    lipsync_resize: int
    max_speakers: int
    realtime_factor: float
    eta_per_minute: str
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def recommend(hw: Optional[HardwareInfo] = None, *, nfe_step: Optional[int] = None,
              measure: bool = False) -> Recommendation:
    """Turn detected hardware into concrete pipeline settings."""
    hw = hw or detect(deep=True)
    tier = hw.tier
    notes: List[str] = []

    d = hw.device
    on_gpu = d in ("cuda", "rocm", "mps", "xpu")
    vram = hw.best_gpu.vram_gb if hw.best_gpu else 0.0

    # ---- defaults per tier ----------------------------------------------
    if tier == "gpu-strong":
        whisper, ctype, steps, f5, lip, resize, spk = "large-v3", "float16", 32, "F5TTS_v1_Base", True, 1, 8
    elif tier == "gpu-modest":
        whisper, ctype, steps, f5, lip, resize, spk = "medium", "float16", 32, "F5TTS_v1_Base", True, 1, 6
    elif tier == "gpu-small":
        whisper, ctype, steps, f5, lip, resize, spk = "small", "float16", 24, "F5TTS_v1_Base", True, 1, 6
    elif tier == "gpu-apple":
        whisper, ctype, steps, f5, lip, resize, spk = "medium", "float32", 24, "F5TTS_v1_Base", True, 1, 6
    elif tier == "cpu-strong":
        whisper, ctype, steps, f5, lip, resize, spk = "medium", "int8", 24, "F5TTS_v1_Base", True, 2, 6
    elif tier == "cpu-modest":
        # Verified working configuration on the reference i3-4130: Whisper
        # small, F5-TTS v1 Base, 16 steps.  Lip sync stays on because the
        # pipeline already skips it automatically beyond lipsync_max_seconds.
        whisper, ctype, steps, f5, lip, resize, spk = "small", "int8", 16, "F5TTS_v1_Base", True, 2, 6
    else:  # cpu-minimal
        whisper, ctype, steps, f5, lip, resize, spk = "base", "int8", 12, "F5TTS_v1_Small", False, 2, 4
        notes.append("Hardware muito limitado: use '--until translate' para conferir o "
                     "texto antes de gastar horas em síntese de voz.")

    if nfe_step is not None:
        steps = int(nfe_step)

    # ---- CPU compute-type sanity ----------------------------------------
    if not on_gpu and hw.cpu.ct2_compute_types and ctype not in hw.cpu.ct2_compute_types:
        fallback = "int8" if "int8" in hw.cpu.ct2_compute_types else hw.cpu.ct2_compute_types[0]
        notes.append(f"'{ctype}' não é suportado por este CPU; usando '{fallback}'.")
        ctype = fallback

    # ---- threads: physical cores, not logical ---------------------------
    threads = max(1, hw.cpu.physical_cores)
    if hw.cpu.physical_cores and hw.cpu.logical_cores > hw.cpu.physical_cores:
        notes.append(
            f"{hw.cpu.logical_cores} threads lógicos / {hw.cpu.physical_cores} núcleos "
            f"físicos: limitando a {threads} threads (hyper-threading atrapalha kernels densos)."
        )

    # ---- memory pressure -------------------------------------------------
    if hw.memory.pressure == "critical":
        notes.append(
            f"Só {hw.memory.available_gb:.1f} GB de RAM livre de {hw.memory.total_gb:.1f} GB. "
            "Feche outros programas ou use um modelo menor, senão vai trocar para o disco."
        )
        if not on_gpu and steps > 16:
            steps = 16
            notes.append("Reduzi os passos de síntese para 16 por causa da memória.")
    elif hw.memory.pressure == "high":
        notes.append(f"RAM livre baixa ({hw.memory.available_gb:.1f} GB).")

    # ---- disk -------------------------------------------------------------
    if hw.disk_free_gb < 6:
        notes.append(
            f"Apenas {hw.disk_free_gb:.1f} GB livres: os modelos ocupam ~2 GB "
            "(F5-TTS 1,4 GB + Whisper) e podem não caber."
        )

    # ---- lip sync feasibility --------------------------------------------
    if lip and not on_gpu:
        notes.append(
            "Sem GPU o lip sync (Wav2Lip) custa ~1 s por quadro e é pulado "
            "automaticamente acima de 90 s de vídeo."
        )

    speed = estimate_speed(hw, nfe_step=steps, measure=measure)
    rtf = speed["realtime_factor"]
    minute_eta = _fmt_eta(60.0 * rtf)
    if not speed["calibrated"]:
        notes.append("Estimativa de velocidade baseada só no número de núcleos; "
                     "rode 'dublador hw --benchmark' para calibrar.")

    return Recommendation(
        tier=tier, tier_label=TIER_LABEL.get(tier, tier), device=d,
        whisper_model=whisper, whisper_compute_type=ctype, threads=threads,
        nfe_step=steps, f5_model=f5, lipsync=lip, lipsync_resize=resize,
        max_speakers=spk, realtime_factor=rtf, eta_per_minute=minute_eta,
        notes=notes,
    )


def _fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    return f"{seconds / 3600:.1f} h"


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def detect(*, deep: bool = True, refresh: bool = False) -> HardwareInfo:
    """Probe the machine.  Results are cached for a few hours.

    Only a **deep** probe is ever cached.  A shallow probe (no torch import)
    cannot see GPUs, so letting it write the cache would silently downgrade
    every later caller to "no GPU" -- which is exactly the bug this guard
    exists to prevent.
    """
    if not refresh:
        cached = read_json(CACHE_FILE, default=None)
        if (isinstance(cached, dict)
                and cached.get("cache_version") == CACHE_VERSION
                and cached.get("probed_at")
                and time.time() - float(cached["probed_at"]) < CACHE_TTL):
            hw = _from_dict(cached)
            if deep or not hw.torch:
                return hw

    phys, logical = _physical_cores()
    cpu = CPUInfo(
        model=_cpu_model(), physical_cores=phys, logical_cores=logical,
        arch=platform.machine(), ct2_compute_types=_ct2_compute_types(),
    )

    mem = _memory()
    gpus, torch_v, cuda_v, notes = _gpus(deep=deep)

    try:
        du = shutil.disk_usage(str(PROJECT_ROOT))
        free_gb, total_gb = du.free / 1024 ** 3, du.total / 1024 ** 3
    except OSError:
        free_gb = total_gb = 0.0

    hw = HardwareInfo(
        cpu=cpu, memory=mem, gpus=gpus, disk_free_gb=round(free_gb, 2),
        disk_total_gb=round(total_gb, 2),
        os_name=f"{platform.system()} {platform.release()}",
        python=platform.python_version(), torch=torch_v, cuda=cuda_v, notes=notes,
    )
    hw.tier = classify(hw)

    if phys <= 2:
        hw.notes.append("Apenas 2 núcleos físicos: a síntese de voz será muito lenta.")
    if deep:
        write_json(CACHE_FILE, hw.to_dict())
    return hw


def _from_dict(d: Dict[str, Any]) -> HardwareInfo:
    hw = HardwareInfo(
        cpu=CPUInfo(**{k: v for k, v in d.get("cpu", {}).items()
                       if k in CPUInfo.__dataclass_fields__}),
        memory=MemoryInfo(**{k: v for k, v in d.get("memory", {}).items()
                             if k in MemoryInfo.__dataclass_fields__}),
        gpus=[GPUInfo(**{k: v for k, v in g.items() if k in GPUInfo.__dataclass_fields__})
              for g in d.get("gpus", [])],
        disk_free_gb=d.get("disk_free_gb", 0.0),
        disk_total_gb=d.get("disk_total_gb", 0.0),
        os_name=d.get("os_name", ""), python=d.get("python", ""),
        torch=d.get("torch", ""), cuda=d.get("cuda", ""),
        tier=d.get("tier", "unknown"), notes=list(d.get("notes", [])),
        probed_at=d.get("probed_at", time.time()),
    )
    # Recompute rather than trusting the stored value, so tweaks to the
    # classification rules take effect without having to bust the cache.
    hw.tier = classify(hw)
    return hw


def apply_to_config(cfg: Any, *, refresh: bool = False) -> Recommendation:
    """Fill every ``auto`` field of a :class:`DubladorConfig` in place.

    Fields the caller set explicitly are left alone; only ``auto``/``0``
    sentinels are replaced.
    """
    hw = detect(deep=True, refresh=refresh)
    rec = recommend(hw)

    if getattr(cfg, "device", "auto") in ("auto", "", None):
        cfg.device = rec.device
    if not getattr(cfg, "threads", 0):
        cfg.threads = rec.threads
    if getattr(cfg, "whisper_model", "auto") in ("auto", "", None):
        cfg.whisper_model = rec.whisper_model
    if getattr(cfg, "whisper_compute_type", "auto") in ("auto", "", None):
        cfg.whisper_compute_type = rec.whisper_compute_type
    if getattr(cfg, "f5_model", "") in ("auto", "", None):
        cfg.f5_model = rec.f5_model
    if getattr(cfg, "tts_nfe_step", 0) in (0, None):
        cfg.tts_nfe_step = rec.nfe_step
    if not getattr(cfg, "max_speakers", 0):
        cfg.max_speakers = rec.max_speakers
    if getattr(cfg, "lipsync_resize_factor", 1) in (0, None):
        cfg.lipsync_resize_factor = rec.lipsync_resize

    cfg.extra["hardware"] = {
        "tier": rec.tier, "summary": hw.summary, "device": rec.device,
        "realtime_factor": rec.realtime_factor, "eta_per_minute": rec.eta_per_minute,
    }

    for n in rec.notes:
        LOG.info("  [auto] %s", n)
    return rec


def report_lines(hw: Optional[HardwareInfo] = None, rec: Optional[Recommendation] = None) -> List[str]:
    """Human-readable multi-line report, shared by the CLI and the GUI."""
    hw = hw or detect(deep=True)
    rec = rec or recommend(hw)
    L: List[str] = []

    L.append("HARDWARE DETECTADO")
    L.append(f"  CPU          {hw.cpu.model}")
    L.append(f"               {hw.cpu.physical_cores} nucleos fisicos / "
             f"{hw.cpu.logical_cores} threads  ({hw.cpu.arch})")
    if hw.cpu.ct2_compute_types:
        L.append(f"  Inferencia   CTranslate2 suporta: {', '.join(hw.cpu.ct2_compute_types)}")
    L.append(f"  Memoria      {hw.memory.total_gb:.1f} GB total, "
             f"{hw.memory.available_gb:.1f} GB livre ({hw.memory.pressure})")
    for g in hw.gpus:
        if g.available:
            L.append(f"  GPU          {g.name}  {g.vram_gb:.1f} GB VRAM  [{g.backend}]")
        else:
            L.append(f"  GPU          {g.name} - nao utilizavel ({g.driver})")
    if not hw.gpus:
        L.append("  GPU          nenhuma detectada")
    L.append(f"  Disco        {hw.disk_free_gb:.1f} GB livres de {hw.disk_total_gb:.1f} GB")
    L.append(f"  Sistema      {hw.os_name} | Python {hw.python}"
             + (f" | torch {hw.torch}" if hw.torch else ""))
    L.append("")
    L.append(f"CLASSIFICACAO  {rec.tier_label}  ({rec.tier})")
    L.append("")
    L.append("AJUSTES ESCOLHIDOS AUTOMATICAMENTE")
    L.append(f"  dispositivo        {rec.device}")
    L.append(f"  threads            {rec.threads}")
    L.append(f"  Whisper            {rec.whisper_model} / {rec.whisper_compute_type}")
    L.append(f"  modelo de voz      {rec.f5_model}  ({rec.nfe_step} passos)")
    L.append(f"  lip sync           {'sim' if rec.lipsync else 'nao'}"
             + (f" (resize {rec.lipsync_resize})" if rec.lipsync else ""))
    L.append(f"  max. locutores     {rec.max_speakers}")
    L.append(f"  velocidade prevista ~{rec.realtime_factor:.0f}x o tempo real "
             f"-> 1 min de video em ~{rec.eta_per_minute}")
    if rec.notes:
        L.append("")
        L.append("OBSERVACOES")
        for n in rec.notes:
            L.append(f"  - {n}")
    return L
