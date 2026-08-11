"""One-click demo of PDF ingestion and chunking."""

from __future__ import annotations

import sys
from pathlib import Path


# Make "project...." imports work when this file is run directly
# with the VS Code ▶ button.
REPO_ROOT = Path(__file__).resolve().parents[2]

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from project.knowledge.ingestion.ingest import ingest_pdf


# ---------------------------------------------------------
# DEMO SETTINGS
# ---------------------------------------------------------

PDF_PATH = (
    REPO_ROOT
    / "knowledge_sources"
    / "papers"
    / "BendsoeTOSpringer.pdf"
)

OUTPUT_ROOT = (
    REPO_ROOT
    / "artifacts"
    / "knowledge"
    / "papers"
)

PAPER_ID = "bendsoe_sigmund_topology_optimization"

MAX_CHARS_PER_CHUNK = 2000

NUMBER_OF_CHUNKS_TO_SHOW = 3


def main() -> None:
    print("=" * 70)
    print("LITERATURE CHUNKING DEMO")
    print("=" * 70)

    print("\n1. Loading PDF...")
    print("   ", PDF_PATH)

    output_directory, metadata, chunks = ingest_pdf(
        PDF_PATH,
        output_root=OUTPUT_ROOT,
        paper_id=PAPER_ID,
        title="Topology Optimization: Theory, Methods, and Applications",
        authors=[
            "Martin Philip Bendsøe",
            "Ole Sigmund",
        ],
        max_chars=MAX_CHARS_PER_CHUNK,
    )

    print("\n2. PDF successfully extracted.")
    print(f"   Pages in PDF:       {metadata.page_count}")
    print(f"   Pages with text:    {metadata.extracted_page_count}")
    print(f"   Characters:         {metadata.extracted_character_count}")
    print(f"   Chunks created:     {len(chunks)}")

    print("\n3. Example chunks")
    print("-" * 70)

    for chunk in chunks[:NUMBER_OF_CHUNKS_TO_SHOW]:
        print()
        print(
            f"{chunk.chunk_id} "
            f"(pages {chunk.page_start}-{chunk.page_end})"
        )
        print()
        print(chunk.text[:700])

        if len(chunk.text) > 700:
            print("...")

        print("-" * 70)

    print("\n4. Example of what Claude could receive")
    print("=" * 70)

    example_chunks = chunks[:2]

    context = "\n\n".join(
        (
            f"[SOURCE: {chunk.chunk_id}, "
            f"pages {chunk.page_start}-{chunk.page_end}]\n"
            f"{chunk.text}"
        )
        for chunk in example_chunks
    )

    print(
        """
QUESTION:
Explain the relevant topology optimization concepts using only
the supplied source material.

SOURCE MATERIAL:
"""
    )

    print(context)

    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)

    print("\nGenerated files are here:")
    print(output_directory)


if __name__ == "__main__":
    main()