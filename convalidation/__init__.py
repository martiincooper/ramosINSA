"""Syllabus-based INSA Lyon <-> USM course convalidation assistant.

This package extracts the *text* of every INSA Lyon and USM course syllabus,
stores each course sheet individually, builds structured metadata, and uses an
LLM-based reasoning layer (with a deterministic fallback) to propose course
convalidations. The PDF text is treated as the factual source; the agent is the
reasoning layer that decides the final equivalence proposals.
"""

__all__ = [
    "config",
    "pdf_utils",
    "insa_parser",
    "usm_parser",
    "llm_agent",
    "matcher",
    "excel_writer",
    "pipeline",
]
