"""Command-line entry point for the INSA <-> USM convalidation pipeline.

Usage::

    python main.py                       # run the full pipeline
    python main.py --insa "INSA COURSES.pdf" --output Results/out.xlsx

The pipeline extracts the syllabus text of every INSA Lyon and USM course,
stores each course sheet individually, and uses an LLM-based agent (with a
deterministic fallback when ``OPENAI_API_KEY`` is not set) to reason over the
extracted text and propose convalidations in an Excel workbook.
"""
from __future__ import annotations

import argparse

from convalidation import pipeline


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--insa",
        dest="insa_pdf",
        default=None,
        help="Path to the bundled INSA courses PDF (default: 'INSA COURSES.pdf').",
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        default=None,
        help="Path for the generated Excel file (default: Results/convalidation_proposals.xlsx).",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    pipeline.run(
        insa_pdf=args.insa_pdf,
        output_path=args.output_path,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
