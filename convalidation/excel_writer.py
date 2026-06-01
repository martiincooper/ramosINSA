"""Write the 5-sheet convalidation workbook (problem statement section 8)."""
from __future__ import annotations

import os
from typing import List

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

from . import config
from .models import CandidateMatch, INSACourse, Recommendation, USMCourse
from .study_plan import StudyPlanRow

_HEADER_FILL = PatternFill("solid", fgColor="1F4E78")
_HEADER_FONT = Font(bold=True, color="FFFFFF")


def _write_sheet(ws, headers: List[str], rows: List[list]) -> None:
    ws.append(headers)
    for cell in ws[1]:
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(vertical="center")
    for row in rows:
        ws.append(row)
    ws.freeze_panes = "A2"
    _autosize(ws, len(headers))


def _autosize(ws, n_cols: int, max_width: int = 60) -> None:
    for col in range(1, n_cols + 1):
        letter = get_column_letter(col)
        longest = 0
        for cell in ws[letter]:
            value = "" if cell.value is None else str(cell.value)
            longest = max(longest, min(len(value), max_width))
        ws.column_dimensions[letter].width = max(12, longest + 2)
        for cell in ws[letter]:
            cell.alignment = Alignment(wrap_text=True, vertical="top")


def write_workbook(
    usm_courses: List[USMCourse],
    insa_courses: List[INSACourse],
    candidates: List[CandidateMatch],
    recommendations: List[Recommendation],
    study_plan: List[StudyPlanRow],
    output_path: str = None,
) -> str:
    output_path = output_path or config.EXCEL_OUTPUT
    wb = Workbook()

    # Sheet 1 - USM Courses
    ws = wb.active
    ws.title = "USM Courses"
    _write_sheet(
        ws,
        ["code", "name", "SCT credits", "department", "description", "key topics"],
        [
            [c.code, c.title, c.sct_credits, c.department, c.description, c.key_topics()]
            for c in usm_courses
        ],
    )

    # Sheet 2 - INSA Courses
    ws = wb.create_sheet("INSA Courses")
    _write_sheet(
        ws,
        ["code", "name", "ECTS", "year", "semester", "department", "key topics"],
        [
            [c.code, c.title, c.ects, c.year, c.semester, c.department, c.key_topics()]
            for c in insa_courses
        ],
    )

    # Sheet 3 - Candidate Matches
    ws = wb.create_sheet("Candidate Matches")
    _write_sheet(
        ws,
        ["USM course", "INSA course", "similarity score", "ECTS", "notes"],
        [
            [
                f"{c.usm_code}",
                f"{c.insa_code} - {c.insa_title}",
                round(c.similarity, 4),
                c.ects,
                c.notes or c.department,
            ]
            for c in candidates
        ],
    )

    # Sheet 4 - Recommended Convalidations
    ws = wb.create_sheet("Recommended Convalidations")
    _write_sheet(
        ws,
        [
            "USM course",
            "recommended INSA course(s)",
            "combined ECTS",
            "estimated equivalence %",
            "validation status",
            "justification",
        ],
        [
            [
                f"{r.usm_code} - {r.usm_title}",
                _format_insa_list(r),
                r.combined_ects,
                round(r.equivalence * 100, 1),
                r.status,
                r.justification,
            ]
            for r in recommendations
        ],
    )

    # Sheet 5 - Final Proposed Study Plan
    ws = wb.create_sheet("Final Proposed Study Plan")
    _write_sheet(
        ws,
        [
            "semester",
            "INSA courses",
            "total ECTS",
            "departments involved",
            "target USM convalidations",
            "notes",
        ],
        [
            [
                row.semester,
                "\n".join(row.insa_courses),
                row.total_ects,
                "\n".join(row.departments),
                ", ".join(row.target_usm),
                "\n".join(row.warnings),
            ]
            for row in study_plan
        ],
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    wb.save(output_path)
    return output_path


def _format_insa_list(rec: Recommendation) -> str:
    if not rec.insa_codes:
        return "-"
    return "\n".join(
        f"{code} - {title}" for code, title in zip(rec.insa_codes, rec.insa_titles)
    )
