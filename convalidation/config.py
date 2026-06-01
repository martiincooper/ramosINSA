"""Central configuration: paths, source PDFs and the academic rules.

All INSA Lyon selection constraints and USM convalidation requirements from the
problem statement are encoded here as data so the matching logic and the LLM
prompt share a single source of truth.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Repository root (the directory that contains the source PDFs).
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Required output folders (created by the pipeline).
INSA_COURSES_DIR = os.path.join(ROOT_DIR, "INSA_Courses")
USM_COURSES_DIR = os.path.join(ROOT_DIR, "USM_Courses")
EXTRACTED_TEXT_DIR = os.path.join(ROOT_DIR, "Extracted_Text")
RESULTS_DIR = os.path.join(ROOT_DIR, "Results")

OUTPUT_DIRS = [INSA_COURSES_DIR, USM_COURSES_DIR, EXTRACTED_TEXT_DIR, RESULTS_DIR]

# Sub-folders inside Extracted_Text for the per-course .txt files.
INSA_TEXT_DIR = os.path.join(EXTRACTED_TEXT_DIR, "INSA")
USM_TEXT_DIR = os.path.join(EXTRACTED_TEXT_DIR, "USM")

EXCEL_OUTPUT = os.path.join(RESULTS_DIR, "convalidation_proposals.xlsx")

# The single PDF that bundles every INSA Lyon course sheet.
INSA_SOURCE_PDF = os.path.join(ROOT_DIR, "INSA COURSES.pdf")

# USM syllabi are individual "PROGRAMA DE ASIGNATURA" PDFs. Any PDF in the
# repository root that is not the INSA bundle is treated as a USM syllabus.
def discover_usm_pdfs() -> List[str]:
    """Return the list of USM syllabus PDFs found in the repository root."""
    pdfs = []
    insa_name = os.path.basename(INSA_SOURCE_PDF)
    for name in sorted(os.listdir(ROOT_DIR)):
        if not name.lower().endswith(".pdf"):
            continue
        if name == insa_name:
            continue
        pdfs.append(os.path.join(ROOT_DIR, name))
    return pdfs


# --------------------------------------------------------------------------- #
# INSA department codes (prefix of the INSA course CODE field, e.g. GI-3-S1-...)
# --------------------------------------------------------------------------- #
INSA_DEPARTMENTS: Dict[str, str] = {
    "GI": "Genie Industriel / Industrial Engineering",
    "HU": "Humanites / Humanities",
    "CDS": "Centre des Sciences / Cross-disciplinary",
    "DD": "Double Diplome / Double Degree",
    "LEAN": "LEAN",
}


def department_name(prefix: str) -> str:
    return INSA_DEPARTMENTS.get(prefix.upper(), prefix.upper())


# --------------------------------------------------------------------------- #
# Academic rules (problem statement sections 1, 2 and 7)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Rules:
    # INSA Lyon constraints (section 1).
    max_ects_per_semester: int = 30
    max_departments: int = 2  # 1 major + 1 minor
    typical_agreement_ects: int = 20
    min_ects_for_internship: int = 15

    # USM convalidation requirements (section 2).
    min_content_equivalence: float = 0.75  # 75 %
    min_ects_per_usm_course: float = 4.5

    # Validation status labels (section 9).
    status_valid: str = "Valid"
    status_borderline: str = "Borderline"
    status_invalid: str = "Invalid"
    status_valid_combination: str = "Valid with combination"
    status_needs_work: str = "Needs additional work"

    # Number of INSA candidates kept per USM course for ranking / LLM input.
    top_k_candidates: int = 8
    # Maximum number of INSA courses combined for a single USM convalidation.
    max_combination_size: int = 3


RULES = Rules()


# Course sheets that are administrative placeholders rather than teachable
# content (academic exchange, internships, gap years, final projects, ...).
# These are excluded from matching even though they appear in INSA COURSES.pdf.
NON_TEACHING_KEYWORDS: List[str] = [
    "academic exchange",
    "echange academique",
    "internship",
    "stage",
    "gap year",
    "cesure",
    "final project",
    "projet de fin",
    "master thesis",
    "no pedagogical activity",
    "sans activite pedagogique",
    "inconnu",
]
