"""Parse USM (UTFSM) ``PROGRAMA DE ASIGNATURA`` syllabi into structured data.

Two layouts appear in the source PDFs and both are supported:

* the current "PROGRAMA DE ASIGNATURA" layout (Asignatura / Sigla / Creditos
  SCT / Departamento / Descripcion / Resultados de Aprendizaje / Contenidos
  tematicos / Metodologia / Bibliografia / cuadro de horas), and
* the legacy layout (ASIGNATURA / SIGLA / CREDITOS / OBJETIVOS / CONTENIDOS /
  BIBLIOGRAFIA).
"""
from __future__ import annotations

import os
import re
from typing import List, Optional, Tuple

from . import config, pdf_utils
from .models import USMCourse

# --------------------------------------------------------------------------- #
# Section markers (ordered list of (field, regex)). Whichever markers are found
# define the section boundaries; a section body runs until the next found marker.
# --------------------------------------------------------------------------- #
_SECTION_MARKERS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("description", re.compile(r"Descripci[oó]n de la [Aa]signatura", re.I)),
    ("objectives", re.compile(r"\bOBJETIVOS\b", re.I)),
    ("entry_requirements", re.compile(r"Requisitos de entrada", re.I)),
    ("profile", re.compile(r"Contribuci[oó]n al perfil de egreso", re.I)),
    ("transversal", re.compile(r"Competencias Transversales", re.I)),
    ("learning_outcomes", re.compile(r"Resultados de Aprendizaje", re.I)),
    ("contents", re.compile(r"Contenidos tem[aá]ticos|\bCONTENIDOS\b", re.I)),
    ("methodology", re.compile(r"Metodolog[ií]a", re.I)),
    ("evaluation", re.compile(r"Evaluaci[oó]n y calificaci[oó]n", re.I)),
    ("bibliography", re.compile(r"Bibliograf[ií]a|Recursos para el aprendizaje", re.I)),
    ("hours_table", re.compile(r"C[AÁ]LCULO DE CANTIDAD DE HORAS|HORAS RELOJ", re.I)),
]


def _split_sections(text: str) -> dict:
    """Return {field: body} using the first occurrence of each section marker."""
    found = []
    for name, pattern in _SECTION_MARKERS:
        m = pattern.search(text)
        if m:
            found.append((m.start(), m.end(), name))
    found.sort()
    sections: dict = {}
    for idx, (_start, end, name) in enumerate(found):
        body_end = found[idx + 1][0] if idx + 1 < len(found) else len(text)
        sections.setdefault(name, text[end:body_end].strip())
    return sections


# --------------------------------------------------------------------------- #
# Identification-block field extraction
# --------------------------------------------------------------------------- #
_SIGLA_RE = re.compile(r"Sigla\s*:?\s*([A-Z]{2,4}\s*-\s*\d{2,3})", re.I)
_SCT_RE = re.compile(r"Cr[eé]ditos\s*SCT\s*:?\s*([0-9]+(?:[.,][0-9]+)?)", re.I)
_UTFSM_RE = re.compile(r"Cr[eé]ditos\s*UTFSM\s*:?\s*([0-9]+(?:[.,][0-9]+)?)", re.I)
_LEGACY_CRED_RE = re.compile(r"CR[EÉ]DITOS\s*:?\s*([0-9]+)", re.I)
_DEPT_RE = re.compile(r"Departamento de\s*\n?\s*([^\n]+)", re.I)
_PREREQ_RE = re.compile(r"Prerr?equisitos?\s*:?\s*\n?([^\n]*)", re.I)
_HOURS_RE = re.compile(
    r"Tiempo total de dedicaci[oó]n a la asignatura\s*:?\s*([0-9]+\s*horas[^.\n]*)", re.I
)
_HOURS_TABLE_RE = re.compile(r"HORAS RELOJ\)\s*([0-9]+(?:[.,][0-9]+)?)", re.I)


def _extract_title(text: str) -> str:
    # Modern layout: "Asignatura: <title> Sigla:" (colon is required so the
    # "ASIGNATURA" inside "PROGRAMA DE ASIGNATURA" / "IDENTIFICACION ... ASIGNATURA."
    # headers is not mistaken for the field label).
    m = re.search(r"Asignatura\s*:\s*(.*?)\s*Sigla\s*:", text, re.I | re.S)
    if not m:
        # Legacy layout: "ASIGNATURA:\n<title>\nSIGLA:"
        m = re.search(r"ASIGNATURA\s*:\s*(.*?)\s*SIGLA\s*:", text, re.I | re.S)
    if not m:
        return ""
    return " ".join(m.group(1).split())


def _extract_semester(text: str) -> str:
    """Read the Impar/Par/Ambos table marking the offered semester with an 'X'."""
    block = re.search(r"Semestre en que se dicta(.{0,80})", text, re.I | re.S)
    if not block:
        return ""
    chunk = block.group(1)
    labels = []
    for label in ("Impar", "Par", "Ambos"):
        # An 'X' near the label (same/next lines) marks the offered semester.
        pat = re.compile(label + r"\s*\n?\s*X", re.I)
        if pat.search(chunk):
            labels.append(label)
    if not labels and "X" in chunk:
        return chunk.strip().split("\n")[0].strip()
    return ", ".join(labels)


def _clean(value: str) -> str:
    return " ".join(value.split())


def _to_float(value: Optional[str]) -> float:
    if not value:
        return 0.0
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return 0.0


def parse_usm_text(text: str, fallback_code: str = "") -> USMCourse:
    sections = _split_sections(text)

    sigla = _SIGLA_RE.search(text)
    code = _clean(sigla.group(1)).replace(" ", "") if sigla else ""
    if not code:
        code = fallback_code

    sct = _SCT_RE.search(text)
    utfsm = _UTFSM_RE.search(text)
    legacy_cred = _LEGACY_CRED_RE.search(text)
    sct_credits = _to_float(sct.group(1)) if sct else _to_float(
        legacy_cred.group(1) if legacy_cred else None
    )

    dept = _DEPT_RE.search(text)
    prereq = _PREREQ_RE.search(text)

    total_hours = ""
    hm = _HOURS_RE.search(text)
    if hm:
        total_hours = _clean(hm.group(1))
    else:
        htab = _HOURS_TABLE_RE.search(text)
        if htab:
            total_hours = htab.group(1) + " horas"

    description = sections.get("description") or sections.get("objectives", "")
    keywords = _build_keywords(text)

    return USMCourse(
        code=code,
        title=_extract_title(text),
        sct_credits=sct_credits,
        utfsm_credits=_to_float(utfsm.group(1)) if utfsm else 0.0,
        department=_clean(dept.group(1)) if dept else "",
        semester=_extract_semester(text),
        prerequisites=_clean(prereq.group(1)) if prereq else "",
        description=_clean(description),
        learning_outcomes=_clean(sections.get("learning_outcomes", "")),
        contents=_clean(sections.get("contents", "")),
        methodology=_clean(sections.get("methodology", "")),
        bibliography=_clean(sections.get("bibliography", "")),
        total_hours=total_hours,
        keywords=keywords,
        raw_text=text,
    )


_KEYWORD_RE = re.compile(r"[A-Za-zÁÉÍÓÚáéíóúÑñ]{4,}")
_STOPWORDS = {
    "para", "como", "esta", "este", "esto", "estos", "estas", "asignatura", "estudiante",
    "donde", "mediante", "través", "traves", "pero", "según", "segun", "sobre", "entre",
    "cada", "tiene", "tiene.", "nivel", "demás", "demas", "otras", "otros", "será", "sera",
}


def _build_keywords(text: str, limit: int = 25) -> str:
    """Most frequent meaningful Spanish words as equivalence-relevant keywords."""
    counts: dict = {}
    for word in _KEYWORD_RE.findall(text.lower()):
        if word in _STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
    return ", ".join(word for word, _ in top)


def parse_usm_pdf(path: str) -> USMCourse:
    """Parse a single USM syllabus PDF and persist its split outputs."""
    text = pdf_utils.read_text(path)
    fallback = pdf_utils.safe_filename(os.path.splitext(os.path.basename(path))[0])
    course = parse_usm_text(text, fallback_code=fallback)

    stem = pdf_utils.safe_filename(course.code or fallback)
    # Copy the (already single-course) syllabus into USM_Courses/ keyed by course.
    course.pdf_path = pdf_utils.split_pdf(
        path,
        list(range(len(pdf_utils.read_pages_text(path)))),
        os.path.join(config.USM_COURSES_DIR, stem + ".pdf"),
    )
    course.text_path = pdf_utils.write_text_file(
        text, os.path.join(config.USM_TEXT_DIR, stem + ".txt")
    )
    return course


def parse_all_usm(paths: Optional[List[str]] = None) -> List[USMCourse]:
    paths = paths if paths is not None else config.discover_usm_pdfs()
    return [parse_usm_pdf(p) for p in paths]
