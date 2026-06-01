"""Low-level PDF helpers: text extraction and page-range splitting."""
from __future__ import annotations

import logging
import os
import re
from typing import List, Sequence

from pypdf import PdfReader, PdfWriter

# pypdf emits a lot of noisy warnings on the malformed source PDFs; silence them.
logging.getLogger("pypdf").setLevel(logging.ERROR)


def read_pages_text(path: str) -> List[str]:
    """Return the extracted text of every page of ``path`` (one item per page)."""
    reader = PdfReader(path)
    pages: List[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # pragma: no cover - defensive against broken pages
            pages.append("")
    return pages


def read_text(path: str) -> str:
    """Return the full extracted text of ``path``."""
    return "\n".join(read_pages_text(path))


def split_pdf(source_path: str, page_indices: Sequence[int], output_path: str) -> str:
    """Write ``page_indices`` of ``source_path`` into a new PDF at ``output_path``."""
    reader = PdfReader(source_path)
    writer = PdfWriter()
    for index in page_indices:
        writer.add_page(reader.pages[index])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as fh:
        writer.write(fh)
    return output_path


def write_text_file(text: str, output_path: str) -> str:
    """Persist ``text`` to ``output_path`` (creating parent folders)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return output_path


_SAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def safe_filename(name: str, max_length: int = 80) -> str:
    """Turn an arbitrary course code/title into a filesystem-safe file stem."""
    cleaned = _SAFE_CHARS.sub("_", name.strip()).strip("_")
    cleaned = cleaned or "course"
    return cleaned[:max_length]
