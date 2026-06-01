"""Build the final proposed study plan (Excel sheet 5) from recommendations.

Aggregates the INSA courses proposed across all *accepted* convalidations and
groups them by INSA semester, reporting total ECTS, departments involved and the
target USM convalidations. INSA constraints are surfaced as warnings (max 30
ECTS / semester, at most 2 departments overall).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .config import RULES
from .models import INSACourse, Recommendation


@dataclass
class StudyPlanRow:
    semester: str
    insa_courses: List[str] = field(default_factory=list)
    total_ects: float = 0.0
    departments: List[str] = field(default_factory=list)
    target_usm: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def build_study_plan(
    recommendations: List[Recommendation], insa_courses: List[INSACourse]
) -> List[StudyPlanRow]:
    insa_by_code = {c.code: c for c in insa_courses}
    buckets: Dict[str, StudyPlanRow] = {}
    seen_per_semester: Dict[str, set] = {}

    for rec in recommendations:
        # The plan is preliminary (problem statement section 1): include every
        # USM course that produced an INSA proposal, regardless of validation
        # status. The per-course status remains visible in the recommendations
        # sheet; Invalid/borderline proposals are still useful for planning.
        if not rec.insa_codes:
            continue
        for code in rec.insa_codes:
            course = insa_by_code.get(code)
            if course is None:
                continue
            semester = f"{course.year} {course.semester}".strip() or "Unscheduled"
            row = buckets.setdefault(semester, StudyPlanRow(semester=semester))
            seen = seen_per_semester.setdefault(semester, set())

            if code not in seen:
                seen.add(code)
                row.insa_courses.append(f"{code} ({course.ects} ECTS)")
                row.total_ects = round(row.total_ects + course.ects, 2)
                if course.department not in row.departments:
                    row.departments.append(course.department)
            if rec.usm_code not in row.target_usm:
                row.target_usm.append(rec.usm_code)

    plan = sorted(buckets.values(), key=lambda r: r.semester)
    _annotate_warnings(plan)
    return plan


def _annotate_warnings(plan: List[StudyPlanRow]) -> None:
    all_departments = set()
    for row in plan:
        all_departments.update(row.departments)
        if row.total_ects > RULES.max_ects_per_semester:
            row.warnings.append(
                f"Exceeds {RULES.max_ects_per_semester} ECTS/semester "
                f"({row.total_ects})."
            )
    if len(all_departments) > RULES.max_departments:
        for row in plan:
            row.warnings.append(
                f"Plan spans {len(all_departments)} departments; INSA allows at "
                f"most {RULES.max_departments} (1 major + 1 minor)."
            )
