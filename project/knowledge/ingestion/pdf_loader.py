"""Deterministic, page-preserving PDF text extraction."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class PdfLoadError(RuntimeError):
    """Raised when a PDF cannot provide usable deterministic text."""


@dataclass(frozen=True, slots=True)
class ExtractedPage:
    """Text extracted from one one-based PDF page."""

    page_number: int
    text: str


@dataclass(frozen=True, slots=True)
class LoadedPdf:
    """Immutable result of page-level PDF extraction."""

    source_filename: str
    source_sha256: str
    page_count: int
    pages: tuple[ExtractedPage, ...]

    @property
    def extracted_page_count(self) -> int:
        """Return the number of pages containing nonblank text."""

        return sum(bool(page.text.strip()) for page in self.pages)

    @property
    def extracted_character_count(self) -> int:
        """Return the total number of extracted characters."""

        return sum(len(page.text) for page in self.pages)


def _sha256_file(path: Path) -> str:
    """Calculate the SHA-256 digest without loading the entire file."""

    digest = sha256()

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _normalize_page_text(text: str) -> str:
    """Normalize platform-dependent line endings while preserving layout."""

    return (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\x00", "")
        .strip()
    )


def load_pdf(pdf_path: str | Path) -> LoadedPdf:
    """Extract a PDF page-by-page and fail if no usable text exists."""

    path = Path(pdf_path)

    if not path.is_file():
        raise FileNotFoundError(f"PDF file does not exist: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file: {path}")

    source_sha256 = _sha256_file(path)

    try:
        reader = PdfReader(str(path))
    except (PdfReadError, OSError, ValueError) as exc:
        raise PdfLoadError(f"Could not read PDF: {path}") from exc

    if reader.is_encrypted:
        try:
            password_result = reader.decrypt("")
        except Exception as exc:
            raise PdfLoadError(
                f"Could not decrypt PDF: {path}"
            ) from exc

        if password_result == 0:
            raise PdfLoadError(
                f"Password-protected PDFs are not supported: {path}"
            )

    page_count = len(reader.pages)

    if page_count == 0:
        raise PdfLoadError(f"PDF contains no pages: {path}")

    extracted_pages: list[ExtractedPage] = []

    for page_number, page in enumerate(reader.pages, start=1):
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            raise PdfLoadError(
                f"Text extraction failed on PDF page {page_number}: {path}"
            ) from exc

        extracted_pages.append(
            ExtractedPage(
                page_number=page_number,
                text=_normalize_page_text(raw_text),
            )
        )

    loaded_pdf = LoadedPdf(
        source_filename=path.name,
        source_sha256=source_sha256,
        page_count=page_count,
        pages=tuple(extracted_pages),
    )

    if loaded_pdf.extracted_character_count == 0:
        raise PdfLoadError(
            "PDF contains no extractable text. "
            "It may be scanned and require OCR, which K1 does not support."
        )

    return loaded_pdf