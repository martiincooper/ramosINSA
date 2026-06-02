"""End-to-end orchestration of the convalidation pipeline."""
from __future__ import annotations

import glob
import os
from typing import List, Optional, Tuple

from . import config, excel_writer, insa_parser, matcher, usm_parser
from .llm_agent import BaseAgent, get_agent
from .models import INSACourse, Recommendation, USMCourse
from .study_plan import build_study_plan


def ensure_output_dirs() -> None:
    for path in config.OUTPUT_DIRS + [config.INSA_TEXT_DIR, config.USM_TEXT_DIR]:
        os.makedirs(path, exist_ok=True)


def load_courses_from_extracted_text() -> Tuple[List[INSACourse], List[USMCourse]]:
    """Build the course models from the already-extracted ``Extracted_Text`` files.

    This skips the (slow, PDF-dependent) extraction/splitting step and reuses the
    per-course ``.txt`` files produced by a previous run. It is the right entry
    point when "the text was already extracted" and only the review/matching step
    is needed.
    """
    insa_courses: List[INSACourse] = []
    for path in sorted(glob.glob(os.path.join(config.INSA_TEXT_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        course = insa_parser.parse_insa_course(text)
        if course is None:
            continue
        course.text_path = path
        insa_courses.append(course)

    usm_courses: List[USMCourse] = []
    seen = set()
    for path in sorted(glob.glob(os.path.join(config.USM_TEXT_DIR, "*.txt"))):
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
        stem = os.path.splitext(os.path.basename(path))[0]
        course = usm_parser.parse_usm_text(text, fallback_code=stem)
        # De-duplicate syllabi extracted under more than one file name.
        key = (course.title.strip().lower(), course.sct_credits)
        if key in seen:
            continue
        seen.add(key)
        course.text_path = path
        usm_courses.append(course)

    return insa_courses, usm_courses


def run(
    insa_pdf: Optional[str] = None,
    agent: Optional[BaseAgent] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
    from_extracted: bool = False,
    manual: bool = False,
) -> str:
    """Run the full pipeline and return the path to the generated Excel file.

    * ``from_extracted=True`` reuses the existing ``Extracted_Text`` files instead
      of re-extracting them from the source PDFs.
    * ``manual=True`` uses the :class:`~convalidation.llm_agent.ManualReviewAgent`
      (the hand-written expert review) instead of an external AI model.
    """
    ensure_output_dirs()
    agent = agent or get_agent(manual=manual)

    def log(message: str) -> None:
        if verbose:
            print(message)

    if from_extracted:
        log("[1/5] Loading INSA and USM courses from existing Extracted_Text ...")
        insa_courses, usm_courses = load_courses_from_extracted_text()
        log(f"      -> {len(insa_courses)} INSA and {len(usm_courses)} USM courses loaded.")
    else:
        log("[1/5] Extracting and splitting INSA course sheets ...")
        insa_courses = insa_parser.parse_insa_pdf(insa_pdf)
        log(f"      -> {len(insa_courses)} INSA course sheets extracted.")

        log("[2/5] Extracting and splitting USM syllabi ...")
        usm_courses = usm_parser.parse_all_usm()
        log(f"      -> {len(usm_courses)} USM courses extracted.")

    log(f"[3/5] Matching with '{agent.name}' reasoning agent ...")
    candidates, recommendations = matcher.match_all(usm_courses, insa_courses, agent)
    log(f"      -> {len(recommendations)} convalidation proposals built.")

    log("[4/5] Building the proposed study plan ...")
    study_plan = build_study_plan(recommendations, insa_courses)

    log("[5/5] Writing the Excel workbook ...")
    path = excel_writer.write_workbook(
        usm_courses,
        insa_courses,
        candidates,
        recommendations,
        study_plan,
        output_path,
    )
    log(f"      -> {path}")

    report_path = write_markdown_report(usm_courses, insa_courses, recommendations)
    log(f"      -> {report_path}")
    return path


def write_markdown_report(
    usm_courses: List[USMCourse],
    insa_courses: List[INSACourse],
    recommendations: List[Recommendation],
    output_path: Optional[str] = None,
) -> str:
    """Write a human-readable Markdown summary of the convalidation proposals."""
    output_path = output_path or os.path.join(
        config.RESULTS_DIR, "convalidation_review.md"
    )
    insa_by_code = {c.code: c for c in insa_courses}
    usm_by_code = {c.code: c for c in usm_courses}

    lines: List[str] = []
    lines.append("# INSA Lyon -> USM convalidation review")
    lines.append("")
    lines.append(
        "Manual, content-based review of the extracted INSA Lyon and USM syllabi. "
        "Equivalence is estimated from the actual syllabus content (not titles); a "
        "convalidation requires >= 75 % content equivalence and >= 4.5 INSA ECTS."
    )
    lines.append("")
    lines.append(
        "| USM course | Recommended INSA course(s) | Combined ECTS | Equivalence | Status |"
    )
    lines.append("| --- | --- | --- | --- | --- |")
    for rec in recommendations:
        usm = usm_by_code.get(rec.usm_code)
        usm_label = f"{rec.usm_code} - {usm.title if usm else rec.usm_title}"
        if rec.insa_codes:
            insa_label = "<br>".join(
                f"{code} - {(insa_by_code[code].title if code in insa_by_code else title)}"
                for code, title in zip(rec.insa_codes, rec.insa_titles)
            )
        else:
            insa_label = "-"
        lines.append(
            f"| {usm_label} | {insa_label} | {rec.combined_ects} | "
            f"{round(rec.equivalence * 100, 1)} % | {rec.status} |"
        )

    lines.append("")
    lines.append("## Justifications")
    lines.append("")
    for rec in recommendations:
        usm = usm_by_code.get(rec.usm_code)
        lines.append(f"### {rec.usm_code} - {usm.title if usm else rec.usm_title}")
        if rec.insa_codes:
            lines.append(
                "- **Proposed INSA course(s):** "
                + ", ".join(
                    f"{code} ({insa_by_code[code].ects} ECTS)"
                    if code in insa_by_code
                    else code
                    for code in rec.insa_codes
                )
            )
        lines.append(f"- **Combined ECTS:** {rec.combined_ects}")
        lines.append(f"- **Estimated equivalence:** {round(rec.equivalence * 100, 1)} %")
        lines.append(f"- **Validation status:** {rec.status}")
        lines.append(f"- **Justification:** {rec.justification}")
        lines.append("")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output_path
