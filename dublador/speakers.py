"""Voice analysis: speaker diarization and F5-TTS reference-clip selection.

Two embedding backends are provided:

``speechbrain``
    ECAPA-TDNN (``speechbrain/spkrec-ecapa-voxceleb``), 192-d, the accurate
    option.  Used automatically when the package is installed.

``mfcc``
    A dependency-free fallback built from librosa: mean+std of MFCCs and
    deltas, pitch statistics and spectral shape.  Weaker than ECAPA but good
    enough to separate a handful of clearly different voices, and it needs no
    model download -- which matters on this CPU-only, low-disk host.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MODELS_DIR
from .media import convert_audio, slice_audio
from .schema import Segment, SpeakerProfile, Transcript
from .utils import LOG, Progress, banner, ensure_dir

TARGET_SR = 16000


# --------------------------------------------------------------------------
# audio loading
# --------------------------------------------------------------------------
def load_mono(path: Path, sr: int = TARGET_SR) -> Tuple[np.ndarray, int]:
    import librosa  # noqa: PLC0415

    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return np.asarray(y, dtype=np.float32), sr


def _trim_silence(y: np.ndarray, sr: int, top_db: float = 32.0) -> np.ndarray:
    import librosa  # noqa: PLC0415

    if y.size == 0:
        return y
    try:
        yt, _ = librosa.effects.trim(y, top_db=top_db)
        return yt if yt.size > sr // 10 else y
    except Exception:  # noqa: BLE001
        return y


# --------------------------------------------------------------------------
# embeddings
# --------------------------------------------------------------------------
_SPEAKER_MODEL = None


def _speechbrain_available() -> bool:
    try:
        import speechbrain  # noqa: F401,PLC0415

        return True
    except Exception:  # noqa: BLE001
        return False


def _load_speechbrain(device: str = "cpu"):
    global _SPEAKER_MODEL
    if _SPEAKER_MODEL is not None:
        return _SPEAKER_MODEL
    from speechbrain.inference.speaker import EncoderClassifier  # noqa: PLC0415

    savedir = MODELS_DIR / "speechbrain-ecapa"
    savedir.mkdir(parents=True, exist_ok=True)
    LOG.info("loading ECAPA-TDNN speaker encoder (%s)", savedir)
    _SPEAKER_MODEL = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir=str(savedir),
        run_opts={"device": device},
    )
    return _SPEAKER_MODEL


def choose_backend(prefer: str = "auto") -> str:
    if prefer != "auto":
        return prefer
    return "speechbrain" if _speechbrain_available() else "mfcc"


def embed_clip(y: np.ndarray, sr: int, backend: str, device: str = "cpu") -> Optional[np.ndarray]:
    """Return an L2-normalised speaker embedding for one utterance."""
    y = _trim_silence(np.asarray(y, dtype=np.float32), sr)
    if y.size < int(0.25 * sr):
        return None

    if backend == "speechbrain":
        try:
            import torch  # noqa: PLC0415

            model = _load_speechbrain(device)
            with torch.no_grad():
                wav = torch.from_numpy(y).unsqueeze(0).float()
                emb = model.encode_batch(wav).squeeze().detach().cpu().numpy()
            emb = np.asarray(emb, dtype=np.float64).ravel()
            return _l2(emb)
        except Exception as e:  # noqa: BLE001
            LOG.debug("speechbrain embedding failed (%s); falling back to mfcc", e)

    return _l2(_mfcc_features(y, sr))


def _l2(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _mfcc_features(y: np.ndarray, sr: int) -> np.ndarray:
    """Compact voice fingerprint: timbre (MFCC) + prosody (F0) + spectrum."""
    import librosa  # noqa: PLC0415

    n_fft = 1024 if len(y) >= 1024 else 512
    hop = 256

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, n_fft=n_fft, hop_length=hop)
    # Drop coefficient 0 (overall loudness, not identity) and de-mean each
    # coefficient so channel/level differences do not dominate the distance.
    mfcc = mfcc[1:]
    mfcc = mfcc - mfcc.mean(axis=1, keepdims=True)
    d1 = librosa.feature.delta(mfcc)

    feats: List[float] = []
    feats += mfcc.mean(axis=1).tolist()
    feats += mfcc.std(axis=1).tolist()
    # NOTE: scale the array *before* .tolist() -- `list * 0.5` is a TypeError.
    feats += (np.abs(d1).mean(axis=1) * 0.5).tolist()

    # prosody
    f0 = estimate_f0(y, sr)
    voiced = f0[f0 > 0]
    if voiced.size >= 4:
        f0_mean = float(np.mean(voiced))
        feats += [math.log(f0_mean + 1e-6), float(np.std(voiced)) / (f0_mean + 1e-6)]
    else:
        feats += [math.log(150.0), 0.0]

    # spectrum
    cent = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    roll = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=n_fft, hop_length=hop)
    feats += [float(np.mean(cent)) / (sr / 2), float(np.mean(roll)) / (sr / 2)]

    v = np.asarray(feats, dtype=np.float64)
    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    return v


def estimate_f0(y: np.ndarray, sr: int) -> np.ndarray:
    """Cheap YIN-based F0 track (0 where unvoiced)."""
    import librosa  # noqa: PLC0415

    if y.size < 512:
        return np.zeros(1, dtype=np.float32)
    try:
        f0 = librosa.yin(y, fmin=60, fmax=400, sr=sr, frame_length=1024, hop_length=256)
        f0 = np.nan_to_num(np.asarray(f0, dtype=np.float32), nan=0.0)
        # YIN reports fmax at the edges; treat those as unvoiced.
        f0[f0 >= 399.0] = 0.0
        return f0
    except Exception:  # noqa: BLE001
        return np.zeros(1, dtype=np.float32)


# --------------------------------------------------------------------------
# clustering
# --------------------------------------------------------------------------
#: Absolute similarity cuts, used only when the caller demands a fixed
#: threshold (``--speaker-threshold > 0``).  The two embedding spaces live on
#: very different scales: ECAPA-TDNN spreads speakers apart (cosine ~0.2-0.7)
#: while raw MFCC statistics compress every voice into a narrow band (>0.9),
#: because most of the vector encodes shared channel/level information rather
#: than identity.  A single hard-coded value is therefore wrong for one of them,
#: which is why the default is now data-driven (:func:`cluster_auto`).
AUTO_THRESHOLD: Dict[str, float] = {
    "speechbrain": 0.62,
    "mfcc": 0.93,
}

#: Silhouette below this means "the split is not convincing" -> one speaker.
MIN_SILHOUETTE = 0.14


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Z-score each feature dimension across utterances.

    Raw MFCC statistics carry a large constant component shared by every
    speaker (channel, recording level, room).  Removing it exposes genuinely
    speaker-specific variation.  Requires a reasonable number of utterances:
    with only three samples, z-scoring makes all points roughly equidistant and
    *causes* over-splitting, so we skip it.
    """
    if matrix.shape[0] < 5:
        return matrix
    mu = matrix.mean(axis=0, keepdims=True)
    sd = matrix.std(axis=0, keepdims=True)
    sd[sd < 1e-9] = 1.0
    return (matrix - mu) / sd


def resolve_threshold(backend: str, threshold: float) -> Optional[float]:
    """Return a fixed cosine cut, or ``None`` to mean 'decide from the data'."""
    if threshold and threshold > 0:
        return threshold
    return None


def cluster_auto(
    embeddings: np.ndarray,
    *,
    min_speakers: int = 1,
    max_speakers: int = 6,
    min_silhouette: float = MIN_SILHOUETTE,
) -> np.ndarray:
    """Pick the number of speakers from the data instead of a magic number.

    We try every plausible ``k``, score the partition with the silhouette
    coefficient (which is scale-invariant, so it behaves identically for ECAPA
    and for the compressed MFCC space) and keep the best.  If even the best
    split is unconvincing we declare a single speaker, which is the safe answer:
    it produces one cloned voice instead of fragmenting one actor into three.
    """
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n < 3 or max_speakers <= 1:
        return np.zeros(n, dtype=int)

    try:
        from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415
        from sklearn.metrics import silhouette_score  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return np.zeros(n, dtype=int)

    X = embeddings.astype(np.float64)
    hi = max(2, min(max_speakers, n - 1))

    best_k = 1
    best_score = -1.0
    best_labels: Optional[np.ndarray] = None

    for k in range(2, hi + 1):
        labels = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(X)
        if len(set(labels.tolist())) < 2:
            continue
        try:
            score = float(silhouette_score(X, labels, metric="cosine"))
        except Exception:  # noqa: BLE001
            continue
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    LOG.debug("auto clustering: n=%d best_k=%d silhouette=%.3f", n, best_k, best_score)
    if best_labels is None or best_score < min_silhouette:
        LOG.debug("  silhouette %.3f < %.3f -> treating as a single speaker",
                  best_score, min_silhouette)
        return np.zeros(n, dtype=int)

    # Honour an explicit minimum.
    if best_k < min_speakers:
        return AgglomerativeClustering(
            n_clusters=min(min_speakers, n), metric="cosine", linkage="average"
        ).fit_predict(X)
    return np.asarray(best_labels, dtype=int)


def cluster_embeddings(
    embeddings: np.ndarray,
    *,
    threshold: float = 0.62,
    min_speakers: int = 1,
    max_speakers: int = 6,
) -> np.ndarray:
    """Agglomerative cosine clustering with a speaker-count guard rail."""
    n = len(embeddings)
    if n == 0:
        return np.zeros(0, dtype=int)
    if n == 1 or max_speakers <= 1:
        return np.zeros(n, dtype=int)

    try:
        from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415
    except ImportError:  # pragma: no cover
        return np.zeros(n, dtype=int)

    # A cosine-distance threshold of 0 means "identical"; map our 0..1
    # similarity-style knob onto a distance.
    distance_threshold = max(0.05, min(0.95, 1.0 - threshold))
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=distance_threshold,
        metric="cosine",
        linkage="average",
    )
    labels = model.fit_predict(embeddings.astype(np.float64))

    k = len(set(labels.tolist()))
    if k < min_speakers:
        model = AgglomerativeClustering(n_clusters=min(min_speakers, n),
                                        metric="cosine", linkage="average")
        labels = model.fit_predict(embeddings.astype(np.float64))
    elif k > max_speakers:
        model = AgglomerativeClustering(n_clusters=max_speakers,
                                        metric="cosine", linkage="average")
        labels = model.fit_predict(embeddings.astype(np.float64))

    return np.asarray(labels, dtype=int)


# --------------------------------------------------------------------------
# main entry points
# --------------------------------------------------------------------------
def diarize(
    audio_16k: Path,
    transcript: Transcript,
    *,
    backend: str = "auto",
    device: str = "cpu",
    threshold: float = 0.0,
    min_speakers: int = 1,
    max_speakers: int = 6,
    min_clip: float = 0.45,
) -> Transcript:
    """Assign ``segment.speaker`` by clustering per-utterance voice embeddings."""
    banner("VOICE ANALYSIS (speaker detection)")

    if not transcript.segments:
        LOG.warning("no segments to diarize")
        return transcript

    chosen = choose_backend(backend)
    fixed_cut = resolve_threshold(chosen, threshold)
    LOG.info("embedding backend: %s (clustering: %s)", chosen,
             f"fixed cosine cut {fixed_cut:.3f}" if fixed_cut
             else "automatic, chosen from the data")

    y, sr = load_mono(audio_16k, TARGET_SR)
    total = len(y) / sr if sr else 0.0
    LOG.info("audio: %.1fs @ %d Hz", total, sr)

    embeddings: List[Optional[np.ndarray]] = []
    prog = Progress(len(transcript.segments), label="  embedding", every=max(1, len(transcript.segments) // 10 or 1))
    for seg in transcript.segments:
        a = max(0, int(seg.start * sr))
        b = min(len(y), int(seg.end * sr))
        clip = y[a:b] if b > a else np.zeros(0, dtype=np.float32)
        if clip.size < int(min_clip * sr):
            embeddings.append(None)
        else:
            embeddings.append(embed_clip(clip, sr, chosen, device))
        prog.step()

    valid_idx = [i for i, e in enumerate(embeddings) if e is not None]
    if not valid_idx:
        LOG.warning("no usable embeddings; assuming a single speaker")
        for seg in transcript.segments:
            seg.speaker = "SPK_01"
        return transcript

    dims = {embeddings[i].shape[0] for i in valid_idx}  # type: ignore[union-attr]
    if len(dims) > 1:
        # Mixed backends (speechbrain failed mid-run) -> keep the majority dim.
        dim = max(dims, key=lambda d: sum(1 for i in valid_idx if embeddings[i].shape[0] == d))
        valid_idx = [i for i in valid_idx if embeddings[i].shape[0] == dim]

    mat = np.vstack([embeddings[i] for i in valid_idx])  # type: ignore[index]

    # Raw MFCC statistics are dominated by a shared constant component, so the
    # fallback backend benefits from per-job standardisation.  ECAPA embeddings
    # are already trained to be directly comparable.
    mat_for_clustering = mat
    if chosen == "mfcc":
        std = standardize(mat)
        if std is not mat:
            norms = np.linalg.norm(std, axis=1, keepdims=True)
            norms[norms < 1e-9] = 1.0
            mat_for_clustering = std / norms

    if fixed_cut is None:
        labels = cluster_auto(mat_for_clustering,
                              min_speakers=min_speakers, max_speakers=max_speakers)
    else:
        labels = cluster_embeddings(mat_for_clustering, threshold=fixed_cut,
                                    min_speakers=min_speakers, max_speakers=max_speakers)

    # Nearest-centroid assignment for segments too short to embed.
    centroids: Dict[int, np.ndarray] = {}
    for lbl in sorted(set(labels.tolist())):
        sel = mat[labels == lbl]
        centroids[int(lbl)] = _l2(sel.mean(axis=0))

    label_of = {idx: int(lbl) for idx, lbl in zip(valid_idx, labels.tolist())}
    for i, seg in enumerate(transcript.segments):
        if i in label_of:
            seg.speaker = f"SPK_{label_of[i] + 1:02d}"
            seg.speaker_confidence = 1.0
        elif centroids:
            # borrow from the temporally nearest already-labelled neighbour
            best, best_d = None, 1e9
            for j in valid_idx:
                d = abs(transcript.segments[j].start - seg.start)
                if d < best_d:
                    best_d, best = d, label_of[j]
            seg.speaker = f"SPK_{best + 1:02d}"
            seg.speaker_confidence = 0.25
        else:
            seg.speaker = "SPK_01"
            seg.speaker_confidence = 0.0

    counts: Dict[str, Dict[str, float]] = {}
    for seg in transcript.segments:
        c = counts.setdefault(seg.speaker or "?", {"n": 0, "t": 0.0})
        c["n"] += 1
        c["t"] += seg.duration

    LOG.info("detected %d speaker(s):", len(counts))
    for spk in sorted(counts):
        LOG.info("  %s: %d segments, %.1fs of speech",
                 spk, int(counts[spk]["n"]), counts[spk]["t"])

    transcript.extra["diarization"] = {
        "backend": chosen,
        "threshold": threshold,
        "n_speakers": len(counts),
    }
    return transcript


def select_reference_clips(
    audio_16k: Path,
    transcript: Transcript,
    *,
    ref_min: float = 4.0,
    ref_max: float = 12.0,
    out_dir: Optional[Path] = None,
) -> Transcript:
    """Pick, per speaker, the cleanest clip to clone the voice from.

    F5-TTS wants roughly 5-12 s of clean, single-speaker, well-recorded speech.
    We score candidates on duration, ASR confidence and lack of silence, then
    cut the winner straight out of the original 16 kHz track.
    """
    banner("VOICE ANALYSIS (reference clips)")

    out_dir = ensure_dir(out_dir or audio_16k.parent)
    y, sr = load_mono(audio_16k, TARGET_SR)

    by_speaker: Dict[str, List[Segment]] = {}
    for seg in transcript.segments:
        if seg.speaker:
            by_speaker.setdefault(seg.speaker, []).append(seg)

    for spk, segs in sorted(by_speaker.items()):
        target = min(ref_max, max(ref_min, 0.5 * (ref_min + ref_max)))
        best: Optional[Tuple[float, Segment, float, float]] = None

        for seg in segs:
            dur = seg.duration
            if dur < 1.2:
                continue
            # Join with the next segment of the same speaker when short.
            start, end = seg.start, seg.end
            if dur < ref_min:
                for nxt in segs:
                    if nxt.start >= end - 0.05 and (end - start) < ref_max:
                        end = max(end, nxt.end)
                    if (end - start) >= ref_min:
                        break
            dur = end - start
            if dur < 1.2:
                continue

            a, b = max(0, int(start * sr)), min(len(y), int(end * sr))
            if b - a < sr:
                continue
            clip = y[a:b]

            speech_ratio = _speech_ratio(clip, sr)
            score = (
                min(dur, ref_max) * 1.0            # prefer long-ish
                - abs(dur - target) * 0.35          # but close to target
                + speech_ratio * 3.0                # prefer dense speech
                + float(seg.avg_logprob) * 0.4      # ASR certainty
                - float(seg.no_speech_prob) * 2.0   # penalise silence
            )
            if best is None or score > best[0]:
                best = (score, seg, start, end)

        if best is None:
            LOG.warning("  %s: no usable reference clip found", spk)
            continue

        _, seg, start, end = best
        # Clamp to the reference window.
        if end - start > ref_max:
            end = start + ref_max

        prof = transcript.speakers.get(spk) or SpeakerProfile(id=spk)
        prof.segments = [s.id for s in segs]
        prof.total_speech = sum(s.duration for s in segs)
        prof.ref_start, prof.ref_end = start, end

        ref_path = out_dir / f"ref_{spk}.wav"
        slice_audio(audio_16k, ref_path, start, end, sample_rate=TARGET_SR, channels=1)
        prof.ref_audio = str(ref_path)
        prof.ref_text = " ".join(
            (s.text or "").strip() for s in segs
            if s.start >= start - 0.01 and s.end <= end + 0.01
        ).strip() or (seg.text or "").strip()

        a, b = max(0, int(start * sr)), min(len(y), int(end * sr))
        f0 = estimate_f0(y[a:b], sr)
        voiced = f0[f0 > 0]
        prof.mean_pitch_hz = float(np.mean(voiced)) if voiced.size else 0.0
        prof.gender = _guess_gender(prof.mean_pitch_hz)

        # Store a centroid embedding for reporting/diagnostics.
        emb = embed_clip(y[a:b], sr, choose_backend("auto"))
        if emb is not None:
            prof.embedding = [round(float(x), 5) for x in emb[:64]]

        transcript.speakers[spk] = prof
        LOG.info("  %s: ref %.1fs (%.2f-%.2fs), pitch %.0f Hz (%s), %d segments, %.1fs speech",
                 spk, end - start, start, end, prof.mean_pitch_hz,
                 prof.gender or "?", len(segs), prof.total_speech)

    return transcript


def _speech_ratio(clip: np.ndarray, sr: int) -> float:
    """Fraction of 20 ms frames above a relative energy floor."""
    if clip.size < sr // 10:
        return 0.0
    frame = max(1, int(0.02 * sr))
    n = clip.size // frame
    if n == 0:
        return 0.0
    rms = np.sqrt(np.mean(clip[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    if rms.max() <= 1e-9:
        return 0.0
    return float(np.mean(rms > rms.max() * 0.08))


def _guess_gender(f0: float) -> Optional[str]:
    if f0 <= 0:
        return None
    if f0 < 165.0:
        return "male"
    if f0 > 185.0:
        return "female"
    return "neutral"
