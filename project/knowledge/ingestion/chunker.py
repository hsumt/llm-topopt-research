"""Deterministic, page-aware chunking for extracted PDF text."""

from __future__ import annotations

from dataclasses import dataclass
import re
import textwrap

from project.knowledge.ingestion.pdf_loader import LoadedPdf
from project.knowledge.schemas import PaperChunk


DEFAULT_MAX_CHARS = 2000

_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*\n+")
_WHITESPACE = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class _TextUnit:
    """One bounded piece of text associated with one PDF page."""

    page_number: int
    text: str


def _split_page_text(
    page_number: int,
    page_text: str,
    max_chars: int,
) -> list[_TextUnit]:
    """Split one page into normalized, size-bounded paragraph units."""

    units: list[_TextUnit] = []

    for paragraph in _PARAGRAPH_BREAK.split(page_text):
        normalized = _WHITESPACE.sub(" ", paragraph).strip()

        if not normalized:
            continue

        pieces = textwrap.wrap(
            normalized,
            width=max_chars,
            break_long_words=True,
            break_on_hyphens=False,
            replace_whitespace=True,
            drop_whitespace=True,
        )

        units.extend(
            _TextUnit(page_number=page_number, text=piece)
            for piece in pieces
        )

    return units


def chunk_pdf(
    loaded_pdf: LoadedPdf,
    paper_id: str,
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> list[PaperChunk]:
    """Create ordered chunks with stable IDs and page provenance."""

    if not paper_id.strip():
        raise ValueError("paper_id cannot be blank")

    if max_chars <= 0:
        raise ValueError("max_chars must be positive")

    units: list[_TextUnit] = []

    for page in loaded_pdf.pages:
        units.extend(
            _split_page_text(
                page_number=page.page_number,
                page_text=page.text,
                max_chars=max_chars,
            )
        )

    if not units:
        raise ValueError("Loaded PDF contains no text units to chunk")

    chunks: list[PaperChunk] = []
    current_units: list[_TextUnit] = []
    current_length = 0

    def emit_current_chunk() -> None:
        nonlocal current_units, current_length

        if not current_units:
            return

        chunk_index = len(chunks)

        chunks.append(
            PaperChunk(
                paper_id=paper_id,
                chunk_id=f"{paper_id}_chunk_{chunk_index:04d}",
                chunk_index=chunk_index,
                page_start=current_units[0].page_number,
                page_end=current_units[-1].page_number,
                text="\n\n".join(
                    unit.text for unit in current_units
                ),
            )
        )

        current_units = []
        current_length = 0

    for unit in units:
        separator_length = 2 if current_units else 0
        candidate_length = (
            current_length + separator_length + len(unit.text)
        )

        if current_units and candidate_length > max_chars:
            emit_current_chunk()

        separator_length = 2 if current_units else 0
        current_units.append(unit)
        current_length += separator_length + len(unit.text)

    emit_current_chunk()

    return chunks