"""Deterministic lexical retrieval over canonical K1 paper chunks."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Return stable lowercase alphanumeric tokens."""

    normalized = unicodedata.normalize("NFKC", text).casefold()
    return _TOKEN_PATTERN.findall(normalized)


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required K1 artifact is missing: {path}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in K1 artifact: {path}") from exc


def load_paper_artifacts(
    paper_directory: str | Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """Load and minimally cross-check K1 metadata and chunks."""

    root = Path(paper_directory)
    metadata = _read_json(root / "metadata.json")
    chunks = _read_json(root / "chunks.json")

    if not isinstance(metadata, dict) or not isinstance(chunks, list):
        raise RuntimeError("K1 metadata/chunks have unexpected JSON types")

    paper_id = metadata.get("paper_id")
    if not isinstance(paper_id, str) or not paper_id:
        raise RuntimeError("metadata.json has no valid paper_id")

    expected_indices = list(range(len(chunks)))
    actual_indices = [chunk.get("chunk_index") for chunk in chunks]
    if actual_indices != expected_indices:
        raise RuntimeError("chunks.json indices are not consecutive")

    chunk_ids = [chunk.get("chunk_id") for chunk in chunks]
    if any(not isinstance(value, str) or not value for value in chunk_ids):
        raise RuntimeError("At least one K1 chunk has no valid chunk_id")
    if len(chunk_ids) != len(set(chunk_ids)):
        raise RuntimeError("chunks.json contains duplicate chunk IDs")

    for chunk in chunks:
        if chunk.get("paper_id") != paper_id:
            raise RuntimeError("At least one chunk has the wrong paper_id")
        if not isinstance(chunk.get("text"), str) or not chunk["text"].strip():
            raise RuntimeError("At least one chunk has blank text")
        page_start = chunk.get("page_start")
        page_end = chunk.get("page_end")
        if not (
            isinstance(page_start, int)
            and isinstance(page_end, int)
            and 1 <= page_start <= page_end
        ):
            raise RuntimeError("At least one chunk has an invalid page range")

    chunks_bytes = (root / "chunks.json").read_bytes()
    chunks_sha256 = hashlib.sha256(chunks_bytes).hexdigest()
    return metadata, chunks, chunks_sha256


@dataclass(frozen=True)
class SearchHit:
    """One ranked retrieval result with its original Tier-A text."""

    rank: int
    score: float
    paper_id: str
    chunk_id: str
    chunk_index: int
    page_start: int
    page_end: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "score": self.score,
            "paper_id": self.paper_id,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "text": self.text,
        }


class BM25Index:
    """Small, dependency-free BM25 index for one ingested paper."""

    def __init__(
        self,
        chunks: Iterable[dict[str, Any]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0.0:
            raise ValueError("k1 must be positive")
        if not 0.0 <= b <= 1.0:
            raise ValueError("b must be in [0, 1]")

        self.chunks = list(chunks)
        if not self.chunks:
            raise ValueError("Cannot build a retrieval index with no chunks")

        self.k1 = float(k1)
        self.b = float(b)
        self.term_frequencies = [
            Counter(tokenize(chunk["text"]))
            for chunk in self.chunks
        ]
        self.document_lengths = [
            sum(frequencies.values())
            for frequencies in self.term_frequencies
        ]
        self.average_document_length = (
            sum(self.document_lengths) / len(self.document_lengths)
        )

        self.document_frequencies: Counter[str] = Counter()
        for frequencies in self.term_frequencies:
            self.document_frequencies.update(frequencies.keys())

    def search(self, query: str, *, top_k: int = 8) -> list[SearchHit]:
        """Rank chunks deterministically; ties resolve by canonical index."""

        query_terms = list(dict.fromkeys(tokenize(query)))
        if not query_terms:
            raise ValueError("Retrieval query contains no searchable terms")
        if top_k < 1:
            raise ValueError("top_k must be at least 1")

        document_count = len(self.chunks)
        scored: list[tuple[float, dict[str, Any]]] = []

        for chunk, frequencies, document_length in zip(
            self.chunks,
            self.term_frequencies,
            self.document_lengths,
            strict=True,
        ):
            score = 0.0
            length_normalizer = self.k1 * (
                1.0
                - self.b
                + self.b
                * document_length
                / self.average_document_length
            )

            for term in query_terms:
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue

                document_frequency = self.document_frequencies[term]
                inverse_document_frequency = math.log(
                    1.0
                    + (
                        document_count
                        - document_frequency
                        + 0.5
                    )
                    / (document_frequency + 0.5)
                )
                score += inverse_document_frequency * (
                    term_frequency * (self.k1 + 1.0)
                    / (term_frequency + length_normalizer)
                )

            if score > 0.0:
                scored.append((score, chunk))

        scored.sort(
            key=lambda item: (
                -item[0],
                item[1]["chunk_index"],
                item[1]["chunk_id"],
            )
        )

        hits: list[SearchHit] = []
        for rank, (score, chunk) in enumerate(scored[:top_k], start=1):
            hits.append(
                SearchHit(
                    rank=rank,
                    score=round(score, 10),
                    paper_id=chunk["paper_id"],
                    chunk_id=chunk["chunk_id"],
                    chunk_index=chunk["chunk_index"],
                    page_start=chunk["page_start"],
                    page_end=chunk["page_end"],
                    text=chunk["text"],
                )
            )

        return hits


def build_retrieval_packet(
    *,
    query: str,
    chunks_sha256: str,
    corpus_chunk_count: int,
    hits: list[SearchHit],
    k1: float,
    b: float,
) -> dict[str, Any]:
    """Create the auditable deterministic retrieval artifact."""

    return {
        "schema_version": "1.0",
        "evidence_tier": "A",
        "retrieval_method": "bm25_lexical_v1",
        "query": query,
        "query_terms": list(dict.fromkeys(tokenize(query))),
        "parameters": {"k1": k1, "b": b, "top_k": len(hits)},
        "chunks_sha256": chunks_sha256,
        "corpus_chunk_count": corpus_chunk_count,
        "hit_count": len(hits),
        "hits": [hit.to_dict() for hit in hits],
    }


def render_context_packet(query: str, hits: list[SearchHit]) -> str:
    """Render exactly the Tier-A context that may be sent to an LLM."""

    lines = [
        "RETRIEVAL QUERY",
        query,
        "",
        "TIER-A CONTEXT SELECTED FOR THE LLM",
        "===================================",
    ]

    for hit in hits:
        lines.extend(
            [
                "",
                (
                    f"[E{hit.rank}] paper_id={hit.paper_id} "
                    f"chunk_id={hit.chunk_id} "
                    f"pages={hit.page_start}-{hit.page_end} "
                    f"bm25={hit.score:.10f}"
                ),
                hit.text,
            ]
        )

    return "\n".join(lines).rstrip() + "\n"
