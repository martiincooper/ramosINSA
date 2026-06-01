"""Reasoning layer for course matching.

The pipeline treats the *extracted syllabus text* as the factual source and an
agent as the reasoning layer. Two agents are provided:

* :class:`OpenAIAgent` - calls an OpenAI chat model when ``OPENAI_API_KEY`` is
  set, reasoning over the extracted INSA/USM syllabus texts and returning a
  structured equivalence judgement.
* :class:`HeuristicAgent` - a deterministic, dependency-free fallback based on
  multilingual bag-of-words similarity over the same extracted text. It lets the
  whole pipeline run end-to-end (and in CI) without any API key.

Both agents share the same interface so the matcher is agnostic to which one is
used. ``get_agent()`` selects automatically.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional

from .lexicon import canonical, strip_accents
from .models import INSACourse, USMCourse


@dataclass
class AgentJudgement:
    """An agent's verdict for one USM course / INSA combination."""

    insa_codes: List[str]
    equivalence: float  # 0..1 estimated content equivalence
    justification: str


class BaseAgent:
    name = "base"

    def score_pair(self, usm: USMCourse, insa: INSACourse) -> float:
        """Return a 0..1 content-similarity estimate for a single INSA course."""
        raise NotImplementedError

    def judge_combination(
        self, usm: USMCourse, combination: List[INSACourse]
    ) -> AgentJudgement:
        """Return an equivalence judgement for a proposed INSA combination."""
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Deterministic multilingual bag-of-words agent (fallback / offline default)
# --------------------------------------------------------------------------- #
_TOKEN_RE = re.compile(r"[a-z0-9áéíóúñàâçèéêëîïôûü]{3,}", re.I)

# Light multilingual (ES / FR / EN) stopword list so similarity reflects topics.
_STOPWORDS = {
    # Spanish
    "para", "como", "esta", "este", "esto", "estos", "estas", "que", "con", "los",
    "las", "del", "una", "unos", "unas", "por", "más", "mas", "sus", "the", "and",
    "asignatura", "estudiante", "mediante", "través", "traves", "según", "segun",
    "sobre", "entre", "cada", "nivel", "donde", "pero", "son", "ser", "sus",
    # French
    "les", "des", "une", "dans", "pour", "avec", "sur", "est", "aux", "par", "ses",
    "cette", "cours", "etudiant", "étudiant", "etudiants", "étudiants",
    # English
    "this", "that", "with", "from", "for", "are", "course", "student", "students",
    "will", "can", "the", "and", "you", "your",
}


def _tokens(text: str) -> List[str]:
    """Tokenise and normalise onto shared cross-lingual concept tokens.

    Tokens are accent-stripped, stop-words removed, and mapped through the
    multilingual concept lexicon so Spanish USM vocabulary aligns with the
    English/French INSA vocabulary.
    """
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text.lower()):
        word = strip_accents(raw)
        if word in _NORMALIZED_STOPWORDS:
            continue
        tokens.append(canonical(word))
    return tokens


_NORMALIZED_STOPWORDS = {strip_accents(w) for w in _STOPWORDS}


def _vector(text: str) -> Counter:
    return Counter(_tokens(text))


def _cosine(a: Counter, b: Counter) -> float:
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    dot = sum(a[t] * b[t] for t in common)
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class HeuristicAgent(BaseAgent):
    name = "heuristic"

    def score_pair(self, usm: USMCourse, insa: INSACourse) -> float:
        return round(_cosine(_vector(usm.matching_text()), _vector(insa.matching_text())), 4)

    def judge_combination(
        self, usm: USMCourse, combination: List[INSACourse]
    ) -> AgentJudgement:
        combined_text = "\n".join(c.matching_text() for c in combination)
        equivalence = _cosine(_vector(usm.matching_text()), _vector(combined_text))
        # Combining several related sheets covers more of the USM syllabus, so we
        # give a mild, capped boost for additional coherent courses.
        if len(combination) > 1:
            equivalence = min(1.0, equivalence * (1.0 + 0.08 * (len(combination) - 1)))
        codes = [c.code for c in combination]
        overlap = sorted(
            set(_tokens(usm.matching_text())) & set(_tokens(combined_text))
        )
        justification = (
            "Heuristic bag-of-words equivalence over extracted syllabus text. "
            f"Shared topic terms: {', '.join(overlap[:12]) or 'none'}."
        )
        return AgentJudgement(codes, round(equivalence, 4), justification)


# --------------------------------------------------------------------------- #
# OpenAI-backed reasoning agent (used when OPENAI_API_KEY is available)
# --------------------------------------------------------------------------- #
class OpenAIAgent(BaseAgent):
    name = "openai"

    def __init__(self, model: Optional[str] = None) -> None:
        from openai import OpenAI  # imported lazily; optional dependency

        self._client = OpenAI()
        self._model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        # Reuse the heuristic for cheap pre-ranking so we only spend tokens on the
        # final combination judgement.
        self._fallback = HeuristicAgent()

    def score_pair(self, usm: USMCourse, insa: INSACourse) -> float:
        return self._fallback.score_pair(usm, insa)

    def judge_combination(
        self, usm: USMCourse, combination: List[INSACourse]
    ) -> AgentJudgement:
        prompt = self._build_prompt(usm, combination)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an academic convalidation assistant comparing "
                            "INSA Lyon course syllabi with USM (UTFSM) course syllabi. "
                            "Reason only over the provided extracted syllabus text, not "
                            "over titles alone. Reply ONLY with JSON."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = response.choices[0].message.content or "{}"
            data = json.loads(_extract_json(content))
            equivalence = float(data.get("equivalence", 0))
            if equivalence > 1:  # model may answer in percent
                equivalence /= 100.0
            justification = str(data.get("justification", "")).strip()
            codes = [c.code for c in combination]
            return AgentJudgement(codes, round(max(0.0, min(1.0, equivalence)), 4), justification)
        except Exception as exc:  # pragma: no cover - network/parse safety net
            judgement = self._fallback.judge_combination(usm, combination)
            judgement.justification = (
                f"[LLM unavailable: {exc}] " + judgement.justification
            )
            return judgement

    @staticmethod
    def _build_prompt(usm: USMCourse, combination: List[INSACourse]) -> str:
        insa_blocks = []
        for c in combination:
            insa_blocks.append(
                f"INSA {c.code} ({c.ects} ECTS, {c.department}, {c.year} {c.semester})\n"
                f"Title: {c.title}\nAims: {c.aims}\nContent: {c.content}\n"
                f"Prerequisites: {c.prerequisites}"
            )
        return (
            "USM course to convalidate:\n"
            f"Code: {usm.code}\nTitle: {usm.title}\nSCT credits: {usm.sct_credits}\n"
            f"Description: {usm.description}\nLearning outcomes: {usm.learning_outcomes}\n"
            f"Contents: {usm.contents}\n\n"
            "Proposed INSA combination:\n" + "\n\n".join(insa_blocks) + "\n\n"
            "Estimate the content equivalence (0..1) of the INSA combination versus "
            "the USM course based strictly on the syllabus content above. Respond with "
            'JSON: {"equivalence": <0..1>, "justification": "<short reason>"}.'
        )


def _extract_json(text: str) -> str:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return "{}"


def get_agent() -> BaseAgent:
    """Return the OpenAI agent when configured, otherwise the heuristic fallback."""
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIAgent()
        except Exception:  # pragma: no cover - missing openai package, etc.
            pass
    return HeuristicAgent()
