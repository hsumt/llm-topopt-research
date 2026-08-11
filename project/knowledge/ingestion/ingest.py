"""Command-line orchestration for deterministic PDF ingestion."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from project.knowledge.ingestion.chunker import (
    DEFAULT_MAX_CHARS,
    chunk_pdf,
)
from project.knowledge.ingestion.pdf_loader import LoadedPdf, load_pdf
from project.knowledge.schemas import PaperChunk, PaperMetadata


DEFAULT_OUTPUT_ROOT = Path("artifacts/knowledge/papers")
_PAPER_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def _automatic_paper_id(loaded_pdf: LoadedPdf) -> str:
    """Create a content-addressed ID that survives file renaming."""

    return f"paper_{loaded_pdf.source_sha256[:16]}"


def _validate_paper_id(paper_id: str) -> str:
    """Reject IDs that could create unsafe or ambiguous paths."""

    if not _PAPER_ID_PATTERN.fullmatch(paper_id):
        raise ValueError(
            "paper_id must contain only lowercase letters, numbers, "
            "underscores, and hyphens"
        )

    return paper_id


def _write_json(path: Path, payload: Any) -> None:
    """Write deterministic, human-readable UTF-8 JSON."""

    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )

    path.write_text(serialized + "\n", encoding="utf-8")


def _guard_existing_identity(
    output_directory: Path,
    source_sha256: str,
) -> None:
    """Prevent one paper ID from silently overwriting another source."""

    metadata_path = output_directory / "metadata.json"

    if not metadata_path.exists():
        return

    try:
        existing_metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Existing metadata is unreadable: {metadata_path}"
        ) from exc

    if existing_metadata.get("source_sha256") != source_sha256:
        raise RuntimeError(
            "The output paper_id already belongs to a different PDF: "
            f"{output_directory.name}"
        )


def ingest_pdf(
    pdf_path: str | Path,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    paper_id: str | None = None,
    title: str | None = None,
    authors: list[str] | None = None,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> tuple[Path, PaperMetadata, list[PaperChunk]]:
    """Extract, chunk, validate, and serialize one research paper."""

    loaded_pdf = load_pdf(pdf_path)

    resolved_paper_id = _validate_paper_id(
        _automatic_paper_id(loaded_pdf)
        if paper_id is None
        else paper_id
    )

    chunks = chunk_pdf(
        loaded_pdf,
        paper_id=resolved_paper_id,
        max_chars=max_chars,
    )

    metadata = PaperMetadata(
        paper_id=resolved_paper_id,
        source_filename=loaded_pdf.source_filename,
        source_sha256=loaded_pdf.source_sha256,
        title=title,
        authors=authors or [],
        page_count=loaded_pdf.page_count,
        extracted_page_count=loaded_pdf.extracted_page_count,
        extracted_character_count=(
            loaded_pdf.extracted_character_count
        ),
    )

    output_directory = Path(output_root) / resolved_paper_id
    output_directory.mkdir(parents=True, exist_ok=True)

    _guard_existing_identity(
        output_directory,
        loaded_pdf.source_sha256,
    )

    pages_payload = [
        {
            "page_number": page.page_number,
            "text": page.text,
        }
        for page in loaded_pdf.pages
    ]

    chunks_payload = [
        chunk.model_dump(mode="json")
        for chunk in chunks
    ]

    manifest_payload = {
        "schema_version": "1.0",
        "paper_id": resolved_paper_id,
        "source_sha256": loaded_pdf.source_sha256,
        "extractor": "pypdf",
        "chunker": "page_aware_paragraph_greedy_v1",
        "max_chars": max_chars,
        "page_count": loaded_pdf.page_count,
        "extracted_page_count": loaded_pdf.extracted_page_count,
        "chunk_count": len(chunks),
        "artifacts": [
            "metadata.json",
            "pages.json",
            "chunks.json",
            "manifest.json",
        ],
    }

    _write_json(
        output_directory / "metadata.json",
        metadata.model_dump(mode="json"),
    )
    _write_json(output_directory / "pages.json", pages_payload)
    _write_json(output_directory / "chunks.json", chunks_payload)
    _write_json(output_directory / "manifest.json", manifest_payload)

    return output_directory, metadata, chunks


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Deterministically extract and chunk one research-paper PDF."
        )
    )
    parser.add_argument("pdf_path", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument("--paper-id")
    parser.add_argument("--title")
    parser.add_argument(
        "--author",
        action="append",
        default=[],
        help="Paper author; repeat this option for multiple authors.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=DEFAULT_MAX_CHARS,
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()

    output_directory, metadata, chunks = ingest_pdf(
        args.pdf_path,
        output_root=args.output_root,
        paper_id=args.paper_id,
        title=args.title,
        authors=args.author,
        max_chars=args.max_chars,
    )

    print("K1 ingestion: PASS")
    print("paper_id:", metadata.paper_id)
    print("source sha256:", metadata.source_sha256)
    print("pages:", metadata.page_count)
    print("extracted pages:", metadata.extracted_page_count)
    print("characters:", metadata.extracted_character_count)
    print("chunks:", len(chunks))
    print("output directory:", output_directory)


if __name__ == "__main__":
    main()