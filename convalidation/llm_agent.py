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
# Manual review agent (a human/expert review encoded as data)
# --------------------------------------------------------------------------- #
# This agent encodes a *manual* content-based review of every USM course against
# the extracted INSA Lyon syllabus catalogue. It exists so the pipeline can
# produce the convalidation output without calling an external AI model: the
# equivalence verdicts below were written by reviewing the extracted INSA / USM
# texts by hand. Each verdict lists the chosen INSA course code(s), an estimated
# content equivalence (0..1) and a justification. The agent steers the matcher so
# it reproduces exactly these decisions while still honouring the academic rules
# (>= 75 % equivalence, >= 4.5 ECTS, combinations, validation status labels).
@dataclass(frozen=True)
class ManualVerdict:
    """A hand-written equivalence verdict for one USM course."""

    insa_codes: tuple
    equivalence: float
    justification: str


# Keyed by the USM course code (upper-case). Two syllabi have no parsable
# "Sigla" field (the innovation-management sheet and the ICN-357 software sheet,
# whose code is written with an en-dash); they are matched by title instead via
# :func:`manual_verdict_for`.
MANUAL_REVIEW: Dict[str, ManualVerdict] = {
    "ELO-302": ManualVerdict(
        ("GI-5-S1-EC-PRI", "GI-3-S2-EC-ISC"),
        0.66,
        "ELO-302 (Proyectos Electronicos) develops and defends an engineering "
        "project before a commission, integrating technical and economic aspects "
        "(market study, business plan, project/investment evaluation). INSA "
        "GI-5-S1-EC-PRI (Industrial project) is a supervised team project answering "
        "a call for tenders that integrates the technical, economic, legal and "
        "human aspects of a project and is defended competitively, mirroring "
        "ELO-302's project methodology and staged defence. GI-3-S2-EC-ISC adds "
        "business-finance content (income statement, balance sheet, management "
        "ratios) and corporate-structure/forecasting analysis supporting the "
        "business-plan component. The market-analysis/marketing and competitive-"
        "strategy emphasis of ELO-302 is only partially covered, so equivalence "
        "stays below the 75 % threshold (borderline).",
    ),
    "ELO-307": ManualVerdict(
        ("GI-4-S1-EC-INR", "GI-5-S1-EC-RGI"),
        0.80,
        "ELO-307 (Proyecto de Titulacion) is a supervised graduation-project "
        "planning course: define the problem and objectives, build the state of "
        "the art, propose solution alternatives, choose the best one and plan the "
        "work. INSA GI-4-S1-EC-INR (Initiation to scientific research) and "
        "GI-5-S1-EC-RGI (Research in Industrial Engineering) are supervised "
        "research projects that follow exactly these steps (appropriation and "
        "formalisation of the research problem, proposal and development of "
        "solutions, analysis of results and perspectives; RGI also requires "
        "writing a research-article-style report). The research methodology is "
        "strongly equivalent and the two courses together meet the credit "
        "requirement.",
    ),
    "ELO-308": ManualVerdict(
        ("GI-4-S1-EC-PCO", "GI-5-S1-EC-IFU"),
        0.55,
        "ELO-308 (Memoria de Titulacion) is the execution of the graduation "
        "project: implement the proposed solution, test and validate it, draw "
        "conclusions and write the final thesis. At INSA this corresponds to the "
        "Projet de Fin d'Etudes / Diploma thesis (PFE / MAS), which are "
        "administrative placeholders excluded from teachable matching. The closest "
        "teachable courses (Collective project PCO and Industry-of-the-future "
        "project IFU) provide hands-on project realisation but do not constitute an "
        "individual engineering thesis, so content equivalence is well below the "
        "threshold: ELO-308 cannot be properly convalidated from the teachable "
        "INSA catalogue.",
    ),
    # Innovation-management sheet (no parsable Sigla -> matched by title too).
    "36": ManualVerdict(
        ("GI-5-S1-EC-KNM", "GI-4-S1-EC-EIE"),
        0.50,
        "Gestion de la Innovacion covers innovation-management strategy: organising "
        "for innovation, technological innovation, innovation strategy, R&D "
        "direction, networks of innovators and capturing the value of innovation. "
        "The INSA Genie Industriel catalogue has no dedicated innovation-management "
        "course; GI-5-S1-EC-KNM (Knowledge management) and GI-4-S1-EC-EIE "
        "(Industrial ecology and circular economy, from the 'continuous improvement "
        "and innovation' unit) only touch fragments (knowledge capital, continuous "
        "improvement, eco-innovation). Both content equivalence and combined ECTS "
        "are insufficient, so no valid convalidation is proposed.",
    ),
    "ICN-322": ManualVerdict(
        ("GI-5-S1-EC-EVP", "GI-4-S1-EC-ISD", "GI-4-S1-EC-DDD"),
        0.66,
        "ICN-322 (Gestion Estrategica) mixes strategic management (strategic "
        "direction/planning, mission-vision, strategic objectives, Canvas business "
        "model, foresight, external/internal analysis) with management-control and "
        "analytics tools (balanced scorecard / cuadro de mando integral, Business "
        "Intelligence, Data Mining, market segmentation/clustering). The analytics "
        "half maps well to INSA GI-5-S1-EC-EVP (Performance evaluation / dashboards "
        "and indicators, i.e. a balanced scorecard), GI-4-S1-EC-ISD (Introduction "
        "to data science: segmentation, clustering, regression, i.e. data mining) "
        "and GI-4-S1-EC-DDD (Data-driven decision making / data warehousing). The "
        "strategic-direction core (planning models such as DELTA, mission/vision, "
        "Canvas, strategic objectives) has no INSA Genie Industriel equivalent, so "
        "overall equivalence stays just below the 75 % threshold (borderline).",
    ),
    "ICN-345": ManualVerdict(
        ("GI-3-S1-EC-GIN", "GI-5-S1-EC-IFD"),
        0.85,
        "ICN-345 (Administracion de la Produccion) is operations/production "
        "management: demand forecasting, inventory management, aggregate planning, "
        "MRP, shop scheduling and control, capacity, location and plant layout, and "
        "world-class manufacturing. INSA GI-3-S1-EC-GIN (Industrial management) "
        "covers the production-management functions, master data, SOP/MPS/MRP and "
        "inventory/Kanban planning, and GI-5-S1-EC-IFD (Internal supply chain and "
        "facility) covers capacity determination, material handling, warehousing "
        "and facility layout (location/distribution). Together they give strong "
        "(> 75 %) coverage of the ICN-345 contents.",
    ),
    "ICN-370": ManualVerdict(
        ("GI-5-S1-EC-SSC", "GI-5-S1-EC-OTL"),
        0.85,
        "ICN-370 (Logistica Industrial) is industrial logistics / supply-chain "
        "management: SCM macro-processes, network and warehousing design, demand "
        "and inventory planning, transport modes (maritime/air/land), intermodality "
        "and port operations. INSA GI-5-S1-EC-SSC (Strategic supply chain) covers "
        "the SCM strategic framework, network and facility-location design, "
        "demand/inventory planning and transportation networks, and GI-5-S1-EC-OTL "
        "(Transportation and logistics optimisation) covers transport problems "
        "(TSP/VRP), logistic-network design and flow optimisation. Together they "
        "are strongly equivalent (> 75 %).",
    ),
    "ILN-230": ManualVerdict(
        ("GI-4-S2-EC-BCG", "GI-5-S1-EC-ADM"),
        0.50,
        "ILN-230 (Ingenieria Economica) covers financial mathematics, economic-"
        "profitability indicators (NPV/IRR) and economic analysis for investment "
        "decision-making. INSA GI-4-S2-EC-BCG (Budget and control management) deals "
        "with financial flows, business-creation finance and performance "
        "indicators, and GI-5-S1-EC-ADM (Multicriteria decision aid) supports "
        "rational decision-making, but neither develops the financial-mathematics / "
        "investment-appraisal (NPV, IRR) core of engineering economics. Equivalence "
        "and combined ECTS are insufficient, so no valid convalidation is proposed.",
    ),
    "ILN-250": ManualVerdict(
        ("GI-3-S1-EC-ROO", "GI-4-S1-EC-OEA"),
        0.85,
        "ILN-250 (Gestion de Investigacion de Operaciones) is operations research: "
        "linear programming, integer programming, non-linear programming and an "
        "introduction to Markov chains and queueing. INSA GI-3-S1-EC-ROO "
        "(Operational research and optimisation) covers LP, the simplex method, "
        "duality/sensitivity, integer programming and branch & bound with modelling "
        "in Excel Solver, and GI-4-S1-EC-OEA (Exact and approached optimisation) "
        "reinforces LP/ILP and adds metaheuristics with an applied optimisation "
        "project. The core OR/optimisation content is strongly equivalent (> 75 %); "
        "only the stochastic part (Markov/queueing) is lightly covered.",
    ),
    # ICN-357 software sheet (its Sigla uses an en-dash so the code is not parsed;
    # matched by title / fallback code via manual_verdict_for).
    "ICN-357": ManualVerdict(
        ("GI-3-S2-EC-CBD", "GI-4-S1-EC-BIN"),
        0.80,
        "ICN-357 (Uso de Software de Ingenieria Industrial) teaches relational "
        "database management (RDBMS): entity-relationship modelling, "
        "normalisation, the relational model and data dictionary, SQL DDL/DML "
        "(create/edit/query), joins/aggregations, stored procedures/triggers, plus "
        "ETL and business-intelligence/data analysis with Python. INSA "
        "GI-3-S2-EC-CBD (Database design and information-systems architecture) "
        "covers the entity-association model, functional dependencies, normal "
        "forms, relational algebra and SQL, and GI-4-S1-EC-BIN (Business "
        "intelligence) covers dimensional data-warehouse modelling, the Extract-"
        "Transform-Load (ETL) task and reporting/scorecards. Together they give "
        "strong (> 75 %) coverage of the database and BI content (the Python-"
        "scripting specifics are only partially covered).",
    ),
}


def manual_verdict_for(usm: USMCourse) -> Optional[ManualVerdict]:
    """Return the manual verdict for a USM course, or ``None`` if not reviewed.

    Matching is primarily by course code; the two syllabi without a parsable code
    (innovation management and ICN-357) are matched by title / raw-text instead.
    """
    code = (usm.code or "").upper()
    if code in MANUAL_REVIEW:
        return MANUAL_REVIEW[code]
    title = (usm.title or "").lower()
    if "innovaci" in title:
        return MANUAL_REVIEW["36"]
    raw = (usm.raw_text or "").upper()
    # Accent-insensitive substring ("ingenier" rather than "ingeniería") so the
    # match is robust to how the accented title was extracted from the PDF.
    if "software de ingenier" in title or "ICN-357" in raw or "ICN\u2013357" in raw:
        return MANUAL_REVIEW["ICN-357"]
    return None


class ManualReviewAgent(BaseAgent):
    """Reproduce the hand-written :data:`MANUAL_REVIEW` verdicts in the matcher.

    The agent gives the manually selected INSA courses the top similarity so they
    are ranked first, and returns the curated equivalence/justification for the
    exact chosen combination. Any other combination is damped below the curated
    value so the matcher always selects the manual proposal, while USM courses not
    covered by the manual review fall back to the deterministic heuristic.
    """

    name = "manual-review"
    # Top similarity for the manually chosen INSA courses so they always rank
    # first and survive the matcher's top-k pruning, ensuring the curated
    # combination is the one selected.
    _CHOSEN_SCORE = 0.97

    def __init__(self) -> None:
        self._fallback = HeuristicAgent()

    def score_pair(self, usm: USMCourse, insa: INSACourse) -> float:
        verdict = manual_verdict_for(usm)
        if verdict and insa.code in verdict.insa_codes:
            return self._CHOSEN_SCORE
        return self._fallback.score_pair(usm, insa)

    def judge_combination(
        self, usm: USMCourse, combination: List[INSACourse]
    ) -> AgentJudgement:
        verdict = manual_verdict_for(usm)
        codes = [c.code for c in combination]
        if verdict is None:
            return self._fallback.judge_combination(usm, combination)

        chosen = set(verdict.insa_codes)
        if set(codes) == chosen:
            return AgentJudgement(codes, verdict.equivalence, verdict.justification)

        # Any partial subset or alternative combination is kept strictly below the
        # curated equivalence so the matcher converges on the full manual proposal.
        base = self._fallback.judge_combination(usm, combination)
        damped = round(min(base.equivalence, verdict.equivalence) * 0.5, 4)
        if set(codes).issubset(chosen):
            note = "Partial subset of the manual convalidation; see the full proposal."
        else:
            note = "Outside the manual convalidation set; non-recommended combination."
        return AgentJudgement(codes, damped, note)


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


def get_agent(manual: bool = False) -> BaseAgent:
    """Return the reasoning agent to use.

    * ``manual=True`` selects the :class:`ManualReviewAgent` (the hand-written
      expert review of the extracted texts), bypassing any external AI model.
    * otherwise the OpenAI agent is used when ``OPENAI_API_KEY`` is configured,
      falling back to the deterministic :class:`HeuristicAgent`.
    """
    if manual:
        return ManualReviewAgent()
    if os.environ.get("OPENAI_API_KEY"):
        try:
            return OpenAIAgent()
        except Exception:  # pragma: no cover - missing openai package, etc.
            pass
    return HeuristicAgent()
