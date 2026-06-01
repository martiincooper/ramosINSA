"""Parse the bundled ``INSA COURSES.pdf`` into individual course sheets.

The bundle starts with a structural index (year / semester / UE / EC listing).
The individual course sheets begin once a page exposes a ``CODE :`` field and
follow a fixed section layout::

    <header / title>
    IDENTIFICATION
    CODE : GI-3-S1-EC-APS
    ECTS : 3
    HOURS
    ...
    ASSESMENT METHOD
    TEACHING AIDS
    TEACHING LANGUAGE
    CONTACT
    AIMS
    CONTENT
    BIBLIOGRAPHY
    PRE-REQUISITES

A course sheet spans the page that carries its ``CODE`` plus any following pages
until the next ``CODE`` page (some sheets overflow onto a second page).
"""
from __future__ import annotations

import os
import re
from typing import List, Optional

from . import config, pdf_utils
from .models import INSACourse

_CODE_RE = re.compile(r"CODE\s*:\s*([A-Z0-9][A-Z0-9\-]+)")
_ECTS_RE = re.compile(r"ECTS\s*:\s*([0-9]+(?:[.,][0-9]+)?)")

# Ordered list of the section headers as they appear in a sheet. Each section's
# body is everything up to the next header in this list.
_SECTION_HEADERS = [
    ("identification", "IDENTIFICATION"),
    ("hours", "HOURS"),
    ("assessment_method", "ASSESMENT METHOD"),  # note: source spells it "ASSESMENT"
    ("teaching_aids", "TEACHING AIDS"),
    ("teaching_language", "TEACHING LANGUAGE"),
    ("contact", "CONTACT"),
    ("aims", "AIMS"),
    ("content", "CONTENT"),
    ("bibliography", "BIBLIOGRAPHY"),
    ("prerequisites", "PRE-REQUISITES"),
]


def _find_course_page_ranges(pages_text: List[str]) -> List[range]:
    """Return one page range per course sheet (CODE page + continuation pages)."""
    code_pages = [i for i, t in enumerate(pages_text) if _CODE_RE.search(t)]
    ranges: List[range] = []
    for idx, start in enumerate(code_pages):
        end = code_pages[idx + 1] if idx + 1 < len(code_pages) else len(pages_text)
        ranges.append(range(start, end))
    return ranges


def _extract_section(text: str, key: str) -> str:
    """Extract the body of one section using the fixed header ordering."""
    header_map = dict(_SECTION_HEADERS)
    header = header_map[key]
    start = text.find(header)
    if start == -1:
        return ""
    body_start = start + len(header)
    # The next header is whichever known header appears first after this one.
    end = len(text)
    for _, other in _SECTION_HEADERS:
        if other == header:
            continue
        pos = text.find(other, body_start)
        if pos != -1:
            end = min(end, pos)
    return text[body_start:end].strip()


def _title_from_text(text: str) -> str:
    """The course title is the text just before the IDENTIFICATION marker.

    INSA repeats the title twice (e.g. "Automated production systems" x2), so we
    de-duplicate the repeated half when present.
    """
    head = text.split("IDENTIFICATION", 1)[0].strip()
    lines = [ln.strip() for ln in head.splitlines() if ln.strip()]
    if not lines:
        return ""
    title = lines[-1]
    half = len(title) // 2
    if half and title[:half].strip() == title[half:].strip():
        title = title[:half].strip()
    return title


def parse_insa_course(text: str) -> Optional[INSACourse]:
    code_match = _CODE_RE.search(text)
    if not code_match:
        return None
    code = code_match.group(1).strip()

    parts = code.split("-")
    dep_prefix = parts[0] if parts else ""
    year = parts[1] if len(parts) > 1 else ""
    semester = parts[2] if len(parts) > 2 else ""

    ects = 0.0
    ects_match = _ECTS_RE.search(text)
    if ects_match:
        ects = float(ects_match.group(1).replace(",", "."))

    return INSACourse(
        code=code,
        title=_title_from_text(text),
        department=config.department_name(dep_prefix),
        year=year,
        semester=semester,
        ects=ects,
        contact_hours=_extract_section(text, "hours"),
        teaching_language=_extract_section(text, "teaching_language"),
        assessment_method=_extract_section(text, "assessment_method"),
        prerequisites=_extract_section(text, "prerequisites"),
        aims=_extract_section(text, "aims"),
        content=_extract_section(text, "content"),
        bibliography=_extract_section(text, "bibliography"),
        contact=_extract_section(text, "contact"),
        notes=_extract_section(text, "teaching_aids"),
        raw_text=text,
    )


def parse_insa_pdf(source_pdf: Optional[str] = None) -> List[INSACourse]:
    """Split ``INSA COURSES.pdf`` and return one :class:`INSACourse` per sheet.

    Each sheet is also written out as an individual PDF (``INSA_Courses/``) and
    text file (``Extracted_Text/INSA/``).
    """
    source_pdf = source_pdf or config.INSA_SOURCE_PDF
    pages_text = pdf_utils.read_pages_text(source_pdf)
    courses: List[INSACourse] = []
    seen_stems: dict = {}

    for page_range in _find_course_page_ranges(pages_text):
        text = "\n".join(pages_text[i] for i in page_range)
        course = parse_insa_course(text)
        if course is None:
            continue

        stem = pdf_utils.safe_filename(course.code or course.title)
        # Guarantee uniqueness if two sheets share a code.
        count = seen_stems.get(stem, 0)
        seen_stems[stem] = count + 1
        if count:
            stem = f"{stem}_{count + 1}"

        course.pdf_path = pdf_utils.split_pdf(
            source_pdf, list(page_range), os.path.join(config.INSA_COURSES_DIR, stem + ".pdf")
        )
        course.text_path = pdf_utils.write_text_file(
            text, os.path.join(config.INSA_TEXT_DIR, stem + ".txt")
        )
        courses.append(course)

    return courses
