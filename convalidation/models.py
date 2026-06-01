"""Structured course models shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


@dataclass
class INSACourse:
    """A single INSA Lyon course sheet (section 5 of the problem statement)."""

    code: str
    title: str
    department: str = ""
    year: str = ""
    semester: str = ""
    ects: float = 0.0
    contact_hours: str = ""
    teaching_language: str = ""
    assessment_method: str = ""
    prerequisites: str = ""
    aims: str = ""
    content: str = ""
    bibliography: str = ""
    contact: str = ""
    notes: str = ""
    raw_text: str = ""
    pdf_path: str = ""
    text_path: str = ""

    def key_topics(self) -> str:
        """Short human-readable summary of the thematic blocks."""
        topics = self.content.strip() or self.aims.strip()
        return _shorten(topics, 300)

    def matching_text(self) -> str:
        """Text fed to the similarity / LLM reasoning layer."""
        return "\n".join(
            part
            for part in (self.title, self.aims, self.content, self.prerequisites, self.bibliography)
            if part
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class USMCourse:
    """A single USM (UTFSM) course syllabus (section 6 of the problem statement)."""

    code: str
    title: str
    sct_credits: float = 0.0
    utfsm_credits: float = 0.0
    department: str = ""
    semester: str = ""
    prerequisites: str = ""
    description: str = ""
    learning_outcomes: str = ""
    contents: str = ""
    methodology: str = ""
    bibliography: str = ""
    total_hours: str = ""
    keywords: str = ""
    raw_text: str = ""
    pdf_path: str = ""
    text_path: str = ""

    def key_topics(self) -> str:
        topics = self.contents.strip() or self.description.strip()
        return _shorten(topics, 300)

    def matching_text(self) -> str:
        return "\n".join(
            part
            for part in (
                self.title,
                self.description,
                self.learning_outcomes,
                self.contents,
                self.keywords,
            )
            if part
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateMatch:
    """One ranked INSA candidate for a given USM course (Excel sheet 3)."""

    usm_code: str
    insa_code: str
    insa_title: str
    similarity: float
    ects: float
    department: str
    notes: str = ""


@dataclass
class Recommendation:
    """A recommended convalidation for one USM course (Excel sheet 4)."""

    usm_code: str
    usm_title: str
    insa_codes: List[str] = field(default_factory=list)
    insa_titles: List[str] = field(default_factory=list)
    combined_ects: float = 0.0
    equivalence: float = 0.0
    status: str = ""
    justification: str = ""
    departments: List[str] = field(default_factory=list)
    semesters: List[str] = field(default_factory=list)


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "\u2026"
