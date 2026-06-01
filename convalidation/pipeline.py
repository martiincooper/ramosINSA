"""End-to-end orchestration of the convalidation pipeline."""
from __future__ import annotations

import os
from typing import Optional

from . import config, excel_writer, insa_parser, matcher, usm_parser
from .llm_agent import BaseAgent, get_agent
from .study_plan import build_study_plan


def ensure_output_dirs() -> None:
    for path in config.OUTPUT_DIRS + [config.INSA_TEXT_DIR, config.USM_TEXT_DIR]:
        os.makedirs(path, exist_ok=True)


def run(
    insa_pdf: Optional[str] = None,
    agent: Optional[BaseAgent] = None,
    output_path: Optional[str] = None,
    verbose: bool = True,
) -> str:
    """Run the full pipeline and return the path to the generated Excel file."""
    ensure_output_dirs()
    agent = agent or get_agent()

    def log(message: str) -> None:
        if verbose:
            print(message)

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
    return path
