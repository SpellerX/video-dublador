"""Naturalise translated dialogue so it sounds spoken, not translated.

A literal, line-by-line machine translation is *correct* but rarely *sayable*.
Professional dubbing fixes that with an adaptation pass, and the problems are
distinct enough to need separate treatments:

1. **Context** -- translating each line alone loses pronouns, gender agreement
   and conversational flow.  Fixed at the source, in :mod:`dublador.translate`
   (windowed translation) and optionally by an LLM here.
2. **Translationese** -- calques, over-formal verb tenses, pronouns that no
   speaker would utter ("Eu irei", "o mesmo", "de forma que").  Fixed by the
   rule packs below.
3. **Register** -- the same line can be *senhor/a* or *cara*; dubbing picks one
   per speaker and keeps it.  Handled by :class:`Register`. 
4. **Isochrony** -- a natural-sounding line that does not fit its time slot is
   useless.  :func:`isochrony_report` estimates spoken duration from the text
   *before* synthesis, so over-long lines are caught in seconds instead of
   after hours of TTS.

The rule packs are deliberately conservative: they only rewrite patterns that
are unambiguously wrong for speech.  Anything subtler needs a language model,
which is why :class:`LLMRewriter` exists as an optional, pluggable stage.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from .schema import Segment, Transcript
from .utils import LOG, Progress, banner, read_json, write_json

# --------------------------------------------------------------------------
# speech rate model (for isochrony)
# --------------------------------------------------------------------------
#: Characters per second of natural dubbing delivery, by language.  Derived
#: from the verified run plus typical dubbing practice: Portuguese needs more
#: characters than English for the same idea, which is why a 1:1 translation
#: so often overflows the slot.
CHARS_PER_SECOND: Dict[str, float] = {
    "en": 15.5, "pt": 14.0, "es": 14.5, "fr": 14.0, "it": 14.5,
    "de": 13.5, "nl": 14.0, "sv": 14.0, "da": 14.0, "no": 14.0,
    "pl": 13.5, "ru": 12.5, "uk": 12.5, "cs": 13.5, "ro": 14.0,
    "tr": 13.5, "fi": 13.0, "hu": 14.0, "el": 13.0,
    "ja": 7.0, "ko": 7.5, "zh": 5.5, "th": 9.0, "vi": 13.0,
    "ar": 12.5, "he": 12.5, "hi": 13.0,
}
DEFAULT_CPS = 14.0


def chars_per_second(lang: str) -> float:
    return CHARS_PER_SECOND.get((lang or "").lower().split("-")[0], DEFAULT_CPS)


def estimate_speech_seconds(text: str, lang: str) -> float:
    """Roughly how long this line takes to say out loud."""
    t = (text or "").strip()
    if not t:
        return 0.0
    cps = chars_per_second(lang)
    # Punctuation implies pauses, so it adds real time beyond the raw count.
    pauses = len(re.findall(r"[,;:]", t)) * 0.18 + len(re.findall(r"[.!?…]", t)) * 0.30
    return len(t) / cps + pauses


# --------------------------------------------------------------------------
# registers (how formal a speaker sounds)
# --------------------------------------------------------------------------
FORMAL_TO_COLLOQUIAL: Dict[str, List[Tuple[str, str]]] = {
    "pt": [
        # ---- futuro do presente -------------------------------------------
        # The single biggest marker of translated Portuguese.  Nobody says
        # "direi a ela amanha"; they say "vou falar pra ela amanha".  We keep
        # the future meaning by using "ir + infinitive" rather than the present
        # tense, which would quietly change what the character is saying.
        (r"\bEu irei\b", "Eu vou"), (r"\bEu farei\b", "Eu vou fazer"),
        (r"\bEu direi\b", "Eu vou dizer"), (r"\bEu poderei\b", "Eu vou poder"),
        (r"\bEu terei\b", "Eu vou ter"), (r"\bEu verei\b", "Eu vou ver"),
        (r"\bEu saberei\b", "Eu vou saber"),
        (r"\bNós iremos\b", "A gente vai"), (r"\bNós vamos\b", "A gente vai"),
        (r"\bnós iremos\b", "a gente vai"), (r"\bnós vamos\b", "a gente vai"),
        # bare forms, for lines where the subject is implied
        (r"\birei\b", "vou"), (r"\bfarei\b", "vou fazer"),
        (r"\bdirei\b", "vou dizer"), (r"\bpoderei\b", "vou poder"),
        (r"\bterei\b", "vou ter"), (r"\bverei\b", "vou ver"),
        (r"\bsaberei\b", "vou saber"), (r"\bconseguirei\b", "vou conseguir"),
        (r"\bquererei\b", "vou querer"), (r"\bprecisarei\b", "vou precisar"),
        (r"\bvoltarei\b", "vou voltar"), (r"\bchegarei\b", "vou chegar"),
        (r"\besperarei\b", "vou esperar"), (r"\btentarei\b", "vou tentar"),
        # pronouns that only exist in written Portuguese
        (r"\ba ela\b", "pra ela"), (r"\ba ele\b", "pra ele"),
        (r"\bpara ela\b", "pra ela"), (r"\bpara ele\b", "pra ele"),
        # contracted speech
        (r"\bEu estou\b", "Eu tô"), (r"\bVocê está\b", "Você tá"),
        (r"\bEle está\b", "Ele tá"), (r"\bEla está\b", "Ela tá"),
        (r"\bEstá bem\b", "Tá bom"), (r"\bEstá certo\b", "Tá certo"),
        (r"\bNão é\?", "Né?"),
        # heavy connectives -> the words people actually say
        (r"\bNo entanto\b", "Mas"), (r"\bno entanto\b", "mas"),
        (r"\bContudo\b", "Mas"), (r"\bTodavia\b", "Mas"),
        (r"\bEntretanto\b", "Mas"), (r"\bentretanto\b", "mas"),
        (r"\bPortanto\b", "Então"), (r"\bportanto\b", "então"),
        (r"\bde forma que\b", "então"), (r"\buma vez que\b", "já que"),
        (r"\bno que diz respeito a\b", "sobre"),
        # stiff verbs
        (r"\bCompreendo\b", "Entendi"), (r"\bCompreendi\b", "Entendi"),
        (r"\bNão compreendo\b", "Não entendi"),
        (r"\bQue é que\b", "O que"),
    ],
    "en": [
        (r"\bI will\b", "I'll"), (r"\bI am\b", "I'm"), (r"\byou are\b", "you're"),
        (r"\bit is\b", "it's"), (r"\bdo not\b", "don't"), (r"\bcannot\b", "can't"),
        (r"\bI shall\b", "I'll"), (r"\bwe will\b", "we'll"),
        (r"\bgoing to\b", "gonna"), (r"\bwant to\b", "wanna"),
        (r"\bkind of\b", "kinda"), (r"\bsort of\b", "sorta"),
        (r"\bthat is\b", "that's"), (r"\bthere is\b", "there's"),
    ],
    "es": [
        (r"\bYo iré\b", "Voy a ir"), (r"\bNo comprendo\b", "No entiendo"),
        (r"\bEstá bien\b", "Vale"),
    ],
}

#: NOTE ON SAFETY
#: --------------
#: Rules that *shift meaning* are deliberately absent, however tempting:
#:   "o mesmo" -> "ele"      breaks "o mesmo aconteceu comigo"
#:                           ("the same thing happened to me")
#:   "com você" -> "contigo" is wrong for Brazilian Portuguese
#:   "Vamos embora" -> "Vamos nessa"  changes "let's leave" into "let's go do it"
#:   "para o" -> "pro"       is a register choice we leave to the writer
#: A naturalisation pass that quietly rewrites the meaning is worse than no
#: pass at all, so those live in TRANSLATIONESE_WARNINGS instead, where they are
#: reported for a human to judge.
SAFE_ONLY = True

#: Interjections and discourse markers that machine translation renders too
#: stiffly.  Applied on the *translated* text, where MT often leaves a literal
#: rendering of the source-language filler.
FILLERS: Dict[str, List[Tuple[str, str]]] = {
    "pt": [
        (r"\bBem,\s*", "Bem, "), (r"\bOra,\s*", "Ora, "),
        (r"\bVocê sabe,\s*", "Sabe, "), (r"\bQuer dizer,\s*", "Quer dizer, "),
        (r"\bOlhe,\s*", "Olha, "), (r"\bEscute,\s*", "Escuta, "),
        (r"\bEi,\s*", "Ei, "), (r"\bEnfim,\s*", "Enfim, "),
        (r"\bDe qualquer forma,\s*", "De qualquer jeito, "),
        (r"\bOK\b", "Tá"), (r"\bOk\b", "Tá"), (r"\bOkay\b", "Tá"),
        (r"\bTudo bem\?", "Tudo bem?"), (r"\bCerto\?", "Certo?"),
        (r"\bSim,\s*", "Sim, "), (r"\bNão,\s*", "Não, "),
    ],
    "en": [
        (r"\bWell,\s*", "Well, "), (r"\bYou know,\s*", "Y'know, "),
        (r"\bI mean,\s*", "I mean, "), (r"\bListen,\s*", "Look, "),
    ],
}

#: Machine-translation artefacts that are always wrong in speech.
GENERIC_CLEANUPS: List[Tuple[str, str]] = [
    (r"\s+([,;:.!?…])", r"\1"),          # space before punctuation
    (r"([,;:.!?…])(?=[^\s\d])", r"\1 "),  # missing space after punctuation
    (r"\s{2,}", " "),                     # collapsed whitespace
    (r"\.{4,}", "..."),
    (r"\s+-\s+", " - "),
    (r'"\s+', '"'), (r'\s+"', '"'),
    (r"\(\s+", "("), (r"\s+\)", ")"),
]

#: Patterns that *look* like literal translation.  Reported, never rewritten:
#: each of these also has a legitimate reading, so guessing a fix would risk
#: changing the meaning -- which is worse than leaving the line alone.
TRANSLATIONESE_WARNINGS: Dict[str, List[Tuple[str, str]]] = {
    "pt": [
        (r"\beventualmente\b",
         "'eventualmente' = 'occasionally', NAO 'eventually' -> use 'por fim'/'finalmente'"),
        (r"\batualmente\b",
         "'atualmente' = 'nowadays', NAO 'actually' -> use 'na verdade'"),
        (r"\bpretender\b",
         "'pretender' = 'intend', NAO 'to pretend' -> use 'fingir'"),
        (r"\brealizar\b",
         "'realizar' como 'perceber' e calque de 'realize' -> use 'perceber'"),
        (r"\bassumir\b",
         "verifique: 'assumir' como 'supor' e calque de 'assume'"),
        (r"\bo mesmo\b",
         "'o mesmo' como pronome (calque de 'the same') soa burocratico -> 'ele'"),
        (r"\bde forma que\b", "conector pesado -> 'entao'/'ai'"),
    ],
    "en": [
        (r"\bthe same\b", "'the same' as a bare pronoun reads stiff"),
        (r"\bI have \d+ years\b", "calque; English uses 'I am N years old'"),
    ],
}


@dataclass
class Register:
    """How a character talks.  ``colloquial`` relaxes formal tenses."""

    name: str = "neutral"           # neutral | colloquial | formal
    glossary: Dict[str, str] = field(default_factory=dict)

    def apply_colloquial(self) -> bool:
        return self.name == "colloquial"


@dataclass
class NaturalizeResult:
    text: str
    original: str
    changed: bool = False
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def compression(self) -> float:
        o = len(self.original)
        return (len(self.text) / o) if o else 1.0


class Naturalizer:
    """Rule-based speech adaptation for one target language."""

    def __init__(self, lang: str, *, glossary: Optional[Dict[str, str]] = None,
                 register: str = "colloquial", aggressive: bool = False):
        self.lang = (lang or "en").lower().split("-")[0]
        self.glossary = dict(glossary or {})
        self.register = normalize_register(register)
        self.aggressive = aggressive
        self._colloquial = FORMAL_TO_COLLOQUIAL.get(self.lang, [])
        self._fillers = FILLERS.get(self.lang, [])
        self._warnings = TRANSLATIONESE_WARNINGS.get(self.lang, [])
        self.stats = {"processed": 0, "changed": 0, "chars_saved": 0}

    # -- one line ----------------------------------------------------------
    def naturalize(self, text: str) -> NaturalizeResult:
        original = text or ""
        t = original.strip()
        if not t:
            return NaturalizeResult("", original)

        notes: List[str] = []

        # 1. glossary first: proper nouns must win over every other rule.
        for src, dst in self.glossary.items():
            if src and src in t:
                t = t.replace(src, dst)

        # 2. colloquial register (the biggest source of "translated" feel)
        if self.register == "colloquial":
            t, hits = _apply_rules_ci(t, self._colloquial)
            if hits:
                notes.append(f"registro: {hits} ajuste(s) de formalidade")

        # 3. discourse markers / interjections
        t, hits = _apply_rules_ci(t, self._fillers)
        if hits:
            notes.append(f"marcadores de fala: {hits}")

        # 4. structural cleanups, always safe
        t, _ = _apply_rules(t, GENERIC_CLEANUPS)

        # 5. flag (never rewrite) literal-translation smell
        warnings = [msg for pat, msg in self._warnings if re.search(pat, t, re.IGNORECASE)]

        # 6. TTS happiness: a terminal punctuation mark avoids flat delivery
        if t and t[-1] not in ".!?…":
            t += "."
        t = re.sub(r"\s{2,}", " ", t).strip()

        self.stats["processed"] += 1
        changed = t != original
        if changed:
            self.stats["changed"] += 1
            self.stats["chars_saved"] += len(original) - len(t)
        return NaturalizeResult(t, original, changed, notes, warnings)

    # -- shorten to fit a slot --------------------------------------------
    def shorten(self, text: str, max_chars: int, *,
                min_keep: float = 0.55, min_chars: int = 14) -> Tuple[str, bool]:
        """Trim a line that is too long to say in the time available.

        Deliberately timid, because an aggressive version of this is far worse
        than the problem it solves: an early implementation turned
        "Eu irei verificar o relatorio amanha" into "Eu vou." and
        "No entanto, eu estou cansado." into "Mas." -- technically fitting the
        slot, and completely destroying the dialogue.

        So a rewrite is accepted only when BOTH hold:

        * it still fits ``max_chars``, and
        * it keeps at least ``min_keep`` of the original characters and
          ``min_chars`` overall.

        If no candidate satisfies that, the text is returned untouched and the
        caller is told nothing was done, so ``fit_to_slot`` can time-compress it
        instead (which preserves meaning, up to its own limit).
        """
        t = (text or "").strip()
        if len(t) <= max_chars:
            return t, False
        # Asking for an impossible length is a signal that the slot is simply
        # too short for this line; mangling the text will not help.
        if max_chars < min_chars:
            return t, False

        floor = max(min_chars, int(len(t) * min_keep))
        candidates: List[str] = []

        # a) drop a trailing vocative / weak clause after a comma
        parts = [p.strip() for p in re.split(r"[,;]", t) if p.strip()]
        if len(parts) > 1:
            candidates.append(", ".join(parts[:-1]) + ".")
            candidates.append(parts[0] + ".")

        # b) drop an explanatory tail introduced by a connector
        for conn in (" porque ", " já que ", " uma vez que ", " embora ",
                     " enquanto ", " para que ", " de modo que "):
            if conn in t:
                candidates.append(t.split(conn)[0].rstrip(",; ") + ".")

        # c) drop a leading discourse marker ("Bem, ..." -> "...")
        candidates.append(re.sub(r"^(Bem|Olha|Escuta|Sabe|Então|Quer dizer|Enfim|Bom),?\s+",
                                 "", t, flags=re.IGNORECASE))

        valid: List[str] = []
        for c in candidates:
            c = c.strip()
            if not c:
                continue
            if c[-1] not in ".!?…":
                c += "."
            if len(c) <= max_chars and len(c) >= floor:
                valid.append(c)

        if not valid:
            return t, False
        return max(valid, key=len), True


def _apply_rules(text: str, rules: Sequence[Tuple[str, str]]) -> Tuple[str, int]:
    hits = 0
    for pat, rep in rules:
        new, n = re.subn(pat, rep, text)
        if n:
            hits += n
            text = new
    return text, hits


def _apply_rules_ci(text: str, rules: Sequence[Tuple[str, str]]) -> Tuple[str, int]:
    """Case-insensitive substitution that preserves the original capitalisation.

    The rule packs are written sentence-initial ("Eu irei" -> "Eu vou"), but the
    same construction appears mid-sentence in lower case ("... disse que eu
    irei").  A case-sensitive pass would silently miss all of those, so we match
    case-insensitively and re-apply the casing of whatever we replaced.
    """
    hits = 0
    for pat, rep in rules:
        try:
            rx = re.compile(pat, re.IGNORECASE)
        except re.error:
            continue

        def do(m: "re.Match[str]", _rep: str = rep) -> str:
            nonlocal hits
            hits += 1
            seg, out = m.group(0), _rep
            # Mirror the capitalisation of what we replaced, in BOTH directions:
            # "No entanto," must become "Mas," (not "mas,"), and a lower-case
            # "eu estou" must not turn into "Eu tô" mid-sentence.
            if seg[:1].isupper() and out[:1].islower():
                out = out[:1].upper() + out[1:]
            elif seg[:1].islower() and out[:1].isupper():
                out = out[:1].lower() + out[1:]
            elif len(seg) > 1 and seg.isupper():
                out = out.upper()
            return out

        text = rx.sub(do, text)
    return text, hits


#: Accepts both the English and the Portuguese spelling, because typing
#: "coloquial" instead of "colloquial" would otherwise disable the whole
#: colloquial pass *silently* -- which is exactly the bug that shipped first.
_REGISTER_ALIASES = {
    "colloquial": "colloquial", "coloquial": "colloquial",
    "informal": "colloquial", "casual": "colloquial",
    "neutral": "neutral", "neutro": "neutral", "padrao": "neutral", "padrão": "neutral",
    "formal": "formal",
}


def normalize_register(value: str) -> str:
    return _REGISTER_ALIASES.get((value or "").strip().lower(), "colloquial")


# --------------------------------------------------------------------------
# transcript level
# --------------------------------------------------------------------------
@dataclass
class IsochronyIssue:
    segment_id: int
    start: float
    slot: float
    estimated: float
    text: str
    overflow: float          # >1 means it will not fit
    max_chars: int


def isochrony_report(tr: Transcript, lang: str, *,
                     tolerance: float = 1.15) -> List[IsochronyIssue]:
    """Flag lines whose translation cannot be spoken in the time available.

    Running this *before* synthesis is the cheap way to catch problems: text is
    free, F5-TTS costs minutes per line.
    """
    issues: List[IsochronyIssue] = []
    for seg in tr.segments:
        text = seg.text_for_tts
        if not text:
            continue
        slot = max(0.3, seg.duration)
        est = estimate_speech_seconds(text, lang)
        if est > slot * tolerance:
            issues.append(IsochronyIssue(
                segment_id=seg.id, start=round(seg.start, 2), slot=round(slot, 2),
                estimated=round(est, 2), text=text,
                overflow=round(est / slot, 2),
                max_chars=int(slot * chars_per_second(lang)),
            ))
    return issues


def naturalize_transcript(
    tr: Transcript,
    lang: str,
    *,
    glossary: Optional[Dict[str, str]] = None,
    register: str = "colloquial",
    fit_slots: bool = True,
    tolerance: float = 1.15,
    shorten_above: float = 1.35,
    llm: Optional["LLMRewriter"] = None,
    out_path: Optional[Any] = None,
) -> Transcript:
    """Rewrite every translated line into natural spoken language.

    ``tolerance`` only controls *reporting*: a line 1.2x too long is fine,
    because ``fit_to_slot`` speeds it up transparently.  ``shorten_above``
    controls *rewriting*: beyond that the time-fitter would start truncating
    speech mid-sentence, so trimming weak material off the line is the lesser
    evil.
    """
    banner("NATURALIZACAO DO TEXTO (adaptacao para dubbing)")

    nat = Naturalizer(lang, glossary=glossary, register=register)
    prog = Progress(len(tr.segments), label="  texto", every=max(1, len(tr.segments) // 10 or 1))

    before = [s.translation or "" for s in tr.segments]

    if llm is not None and llm.available():
        LOG.info("  refinando %d falas com %s", len(tr.segments), llm.describe())
        llm.rewrite_segments(tr, lang, register=register, progress=prog)
    else:
        for seg in tr.segments:
            res = nat.naturalize(seg.text_for_tts)
            if res.changed:
                seg.translation = res.text
            seg.fit_meta.setdefault("naturalize", {})
            seg.fit_meta["naturalize"] = {
                "changed": res.changed, "notes": res.notes, "warnings": res.warnings,
            }
            prog.step()

    # ---- isochrony: make the text fit the picture ------------------------
    issues = isochrony_report(tr, lang, tolerance=tolerance)
    severe = [i for i in issues if i.overflow > shorten_above]
    shortened = 0

    if issues:
        LOG.info("")
        LOG.info("  %d fala(s) mais longas que a janela (de %d no total):",
                 len(issues), len(tr.segments))
        for it in issues[:8]:
            mark = "!" if it.overflow > shorten_above else " "
            LOG.info("   %s seg %-4d janela %.2fs, fala ~%.2fs (%.2fx): %.52s",
                     mark, it.segment_id, it.slot, it.estimated, it.overflow, it.text)
        if len(issues) > 8:
            LOG.info("    ... e mais %d", len(issues) - 8)
        if severe:
            LOG.info("  ('!' = %.2fx ou mais: o ajuste de tempo cortaria a fala, "
                     "entao vale encurtar o texto)", shorten_above)
        modest = len(issues) - len(severe)
        if modest:
            LOG.info("  %d fala(s) entre %.2fx e %.2fx: o ajuste de tempo resolve, "
                     "texto mantido", modest, tolerance, shorten_above)

        if fit_slots and severe:
            by_id = {s.id: s for s in tr.segments}
            for it in severe:
                seg = by_id.get(it.segment_id)
                if not seg:
                    continue
                short, ok = nat.shorten(seg.text_for_tts, it.max_chars)
                if ok and short and len(short) < len(seg.text_for_tts):
                    LOG.debug("    seg %d: %.60s -> %.60s", seg.id, seg.text_for_tts, short)
                    seg.translation = short
                    shortened += 1
            LOG.info("  %d fala(s) encurtada(s) com seguranca", shortened)
            left = len(severe) - shortened
            if left:
                LOG.info("  %d fala(s) longa(s) demais para encurtar sem perder o "
                         "sentido - o ajuste de tempo vai acelerar ate o limite", left)
        elif severe and not fit_slots:
            LOG.info("  (encurtamento desativado; o ajuste de tempo vai acelerar essas falas)")
    else:
        LOG.info("  todas as falas cabem na janela de tempo original")

    # ---- report ----------------------------------------------------------
    changed = sum(1 for s, b in zip(tr.segments, before) if (s.translation or "") != b)
    warn_total = sum(len(s.fit_meta.get("naturalize", {}).get("warnings", []))
                     for s in tr.segments)

    LOG.info("")
    LOG.info("  falas reescritas : %d de %d", changed, len(tr.segments))
    LOG.info("  encurtadas       : %d", shortened)
    LOG.info("  avisos de calque : %d", warn_total)
    if nat.stats["chars_saved"]:
        LOG.info("  caracteres a menos: %d (%.1f%% menor, ajuda a caber no tempo)",
                 nat.stats["chars_saved"],
                 100.0 * nat.stats["chars_saved"] / max(1, sum(len(b) for b in before)))

    tr.extra["naturalize"] = {
        "language": lang, "register": register,
        "changed": changed, "shortened": shortened,
        "isochrony_issues": len(issues), "warnings": warn_total,
        "llm": llm.describe() if (llm and llm.available()) else None,
    }

    if out_path:
        write_json(out_path, [
            {"id": b_id, "before": b, "after": s.translation,
             "notes": s.fit_meta.get("naturalize", {}).get("notes", []),
             "warnings": s.fit_meta.get("naturalize", {}).get("warnings", [])}
            for b_id, b, s in zip([x.id for x in tr.segments], before, tr.segments)
            if (s.translation or "") != b
        ])
    return tr


# --------------------------------------------------------------------------
# optional LLM rewrite (best quality, needs a model)
# --------------------------------------------------------------------------
DEFAULT_LLM_PROMPT = """You adapt subtitles into dubbing scripts.

Rewrite each line as natural SPOKEN {language}, the way a real actor would say
it on screen. Rules:
- Keep the exact meaning. Never add or remove information.
- Match the register of the original ({register}); contractions and everyday
  wording are good.
- Keep each line roughly the same LENGTH as the input, because it must fit the
  original timing. Shorter is better than longer.
- Keep names, numbers and terms from this glossary: {glossary}
- Output ONLY the rewritten lines, one per input line, numbered the same way.
  No explanations, no commentary.

{numbered_lines}"""


class LLMRewriter:
    """Optional LLM pass for genuinely natural dialogue.

    Supports any OpenAI-compatible chat endpoint (OpenAI, Groq, Together,
    OpenRouter, vLLM, LM Studio...) and Ollama's native API.  This is the only
    part of the naturalisation that can fix *subtle* register and idiom
    problems; the rule packs handle the mechanical ones.

    It is entirely optional: with no endpoint configured the rule-based
    naturaliser runs alone and the pipeline still works.
    """

    def __init__(self, base_url: str = "", model: str = "", api_key: str = "",
                 timeout: float = 120.0, batch: int = 25, flavour: str = "openai"):
        self.base_url = (base_url or "").rstrip("/")
        self.model = model or "llama3.1"
        self.api_key = api_key
        self.timeout = timeout
        self.batch = max(1, batch)
        self.flavour = flavour          # openai | ollama

    def available(self) -> bool:
        return bool(self.base_url)

    def describe(self) -> str:
        if not self.available():
            return "desativado"
        return f"{self.flavour} {self.model} @ {self.base_url}"

    # -- transport ---------------------------------------------------------
    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        import urllib.request  # noqa: PLC0415

        url = f"{self.base_url}{path}"
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    def _chat(self, system: str, user: str) -> str:
        if self.flavour == "ollama":
            out = self._post("/api/chat", {
                "model": self.model, "stream": False,
                "messages": [{"role": "system", "content": system},
                             {"role": "user", "content": user}],
                "options": {"temperature": 0.3},
            })
            return str((out.get("message") or {}).get("content") or "")

        out = self._post("/chat/completions", {
            "model": self.model, "temperature": 0.3,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": user}],
        })
        try:
            return str(out["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError):
            return ""

    # -- batching ----------------------------------------------------------
    def rewrite_segments(self, tr: Transcript, lang: str, *,
                         register: str = "colloquial",
                         progress: Optional[Progress] = None) -> Transcript:
        from .config import LANGUAGES  # noqa: PLC0415

        todo = [s for s in tr.segments if s.text_for_tts]
        if not todo:
            return tr

        lang_name = LANGUAGES.get(lang, lang)
        glossary = ", ".join(f"{k}->{v}" for k, v in (tr.extra.get("glossary") or {}).items()) or "(nenhum)"

        for start in range(0, len(todo), self.batch):
            chunk = todo[start:start + self.batch]
            numbered = "\n".join(f"{i + 1}. {s.text_for_tts}" for i, s in enumerate(chunk))
            prompt = DEFAULT_LLM_PROMPT.format(
                language=lang_name, register=register, glossary=glossary,
                numbered_lines=numbered,
            )
            try:
                raw = self._chat(
                    "You are a professional dubbing script adaptor. You output "
                    "only the numbered lines, nothing else.",
                    prompt,
                )
            except Exception as e:  # noqa: BLE001
                LOG.warning("  LLM indisponivel (%s); mantendo o texto das regras", e)
                return tr

            parsed = _parse_numbered(raw, len(chunk))
            for seg, new in zip(chunk, parsed):
                if new and len(new) > 1:
                    seg.fit_meta.setdefault("naturalize", {})
                    seg.fit_meta["naturalize"] = {"changed": True, "notes": ["llm"], "warnings": []}
                    seg.translation = new.strip()
            if progress:
                for _ in chunk:
                    progress.step()
        return tr


def _parse_numbered(raw: str, expected: int) -> List[str]:
    """Recover ``N. text`` lines from an LLM reply, tolerating stray prose."""
    out: List[Optional[str]] = [None] * expected
    for line in (raw or "").splitlines():
        m = re.match(r"^\s*(\d{1,3})\s*[.)\-:]\s*(.+?)\s*$", line)
        if not m:
            continue
        idx = int(m.group(1)) - 1
        if 0 <= idx < expected:
            out[idx] = m.group(2)
    return [o or "" for o in out]
