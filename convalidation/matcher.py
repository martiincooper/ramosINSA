"""Matching logic: rank INSA candidates and build convalidation proposals.

Implements the requirements of the problem statement:

* compare each USM course against all (teachable) INSA courses using the
  extracted syllabus text (not titles),
* rank plausible matches and keep the top candidates,
* prefer combinations of INSA courses when one is insufficient, favouring the
  same department and avoiding unnecessary cross-year / cross-department mixes,
* require >= 75 % content equivalence and >= 4.5 ECTS, and
* assign a validation status label.
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

from . import config
from .config import RULES
from .llm_agent import BaseAgent, get_agent
from .models import CandidateMatch, INSACourse, Recommendation, USMCourse


def is_teachable(course: INSACourse) -> bool:
    """Exclude administrative placeholders (exchange, internship, gap year, ...)."""
    if course.ects <= 0:
        return False
    haystack = f"{course.title} {course.content} {course.aims}".lower()
    return not any(kw in haystack for kw in config.NON_TEACHING_KEYWORDS)


def rank_candidates(
    usm: USMCourse, insa_courses: List[INSACourse], agent: BaseAgent
) -> List[CandidateMatch]:
    """Score and rank every teachable INSA course against ``usm``."""
    scored: List[CandidateMatch] = []
    for insa in insa_courses:
        if not is_teachable(insa):
            continue
        similarity = agent.score_pair(usm, insa)
        scored.append(
            CandidateMatch(
                usm_code=usm.code,
                insa_code=insa.code,
                insa_title=insa.title,
                similarity=similarity,
                ects=insa.ects,
                department=insa.department,
                notes="",
            )
        )
    scored.sort(key=lambda c: c.similarity, reverse=True)
    # De-duplicate by INSA code (the bundle can contain a course twice) keeping
    # the highest-scoring occurrence, so combinations never reuse the same course.
    unique: List[CandidateMatch] = []
    seen_codes = set()
    for candidate in scored:
        if candidate.insa_code in seen_codes:
            continue
        seen_codes.add(candidate.insa_code)
        unique.append(candidate)
    return unique


def _status_for(equivalence: float, ects: float) -> str:
    meets_equiv = equivalence >= RULES.min_content_equivalence
    meets_ects = ects >= RULES.min_ects_per_usm_course
    if meets_equiv and meets_ects:
        return RULES.status_valid
    if meets_equiv and not meets_ects:
        # Strong thematic match but a little short on credits.
        return RULES.status_needs_work
    if not meets_equiv and meets_ects and equivalence >= RULES.min_content_equivalence - 0.1:
        return RULES.status_borderline
    return RULES.status_invalid


def _best_combination(
    usm: USMCourse,
    candidates: List[CandidateMatch],
    insa_by_code: dict,
    agent: BaseAgent,
) -> Optional[Tuple[List[INSACourse], float]]:
    """Search small INSA combinations for one reaching equivalence and ECTS goals.

    Combinations are built from the top candidates, preferring those in the same
    department as the strongest candidate to keep timetable risk and coherence in
    check. Returns the best (courses, equivalence) found, biased toward proposals
    that satisfy both thresholds with the fewest, most coherent courses.
    """
    if not candidates:
        return None

    pool = [insa_by_code[c.insa_code] for c in candidates[: RULES.top_k_candidates]]
    primary_department = pool[0].department

    best: Optional[Tuple[List[INSACourse], float]] = None
    best_score = -1.0

    for size in range(1, RULES.max_combination_size + 1):
        for combo in combinations(pool, size):
            judgement = agent.judge_combination(usm, list(combo))
            equivalence = judgement.equivalence
            ects = sum(c.ects for c in combo)
            departments = {c.department for c in combo}

            meets = (
                equivalence >= RULES.min_content_equivalence
                and ects >= RULES.min_ects_per_usm_course
            )
            # Ranking preference: satisfying both thresholds first, then higher
            # equivalence, then fewer courses, then single-department coherence.
            score = (
                (2.0 if meets else 0.0)
                + equivalence
                - 0.05 * (size - 1)
                - (0.1 if len(departments) > 1 else 0.0)
                - (0.05 if primary_department not in departments else 0.0)
            )
            if score > best_score:
                best_score = score
                best = (list(combo), equivalence)

        # Stop enlarging combinations once a compliant proposal exists.
        if best and best[1] >= RULES.min_content_equivalence:
            combo, _ = best
            if sum(c.ects for c in combo) >= RULES.min_ects_per_usm_course:
                break

    return best


def recommend(
    usm: USMCourse,
    candidates: List[CandidateMatch],
    insa_courses: List[INSACourse],
    agent: BaseAgent,
) -> Recommendation:
    insa_by_code = {c.code: c for c in insa_courses}
    result = _best_combination(usm, candidates, insa_by_code, agent)

    if not result:
        return Recommendation(
            usm_code=usm.code,
            usm_title=usm.title,
            status=RULES.status_invalid,
            justification="No teachable INSA candidate found for this USM course.",
        )

    combo, equivalence = result
    judgement = agent.judge_combination(usm, combo)
    combined_ects = sum(c.ects for c in combo)
    status = _status_for(equivalence, combined_ects)
    if status == RULES.status_valid and len(combo) > 1:
        status = RULES.status_valid_combination

    return Recommendation(
        usm_code=usm.code,
        usm_title=usm.title,
        insa_codes=[c.code for c in combo],
        insa_titles=[c.title for c in combo],
        combined_ects=round(combined_ects, 2),
        equivalence=round(equivalence, 4),
        status=status,
        justification=judgement.justification,
        departments=sorted({c.department for c in combo}),
        semesters=sorted({f"{c.year} {c.semester}".strip() for c in combo}),
    )


def match_all(
    usm_courses: List[USMCourse],
    insa_courses: List[INSACourse],
    agent: Optional[BaseAgent] = None,
) -> Tuple[List[CandidateMatch], List[Recommendation]]:
    """Return (all top candidate matches, one recommendation per USM course)."""
    agent = agent or get_agent()
    all_candidates: List[CandidateMatch] = []
    recommendations: List[Recommendation] = []

    for usm in usm_courses:
        ranked = rank_candidates(usm, insa_courses, agent)
        top = ranked[: RULES.top_k_candidates]
        all_candidates.extend(top)
        recommendations.append(recommend(usm, ranked, insa_courses, agent))

    return all_candidates, recommendations
