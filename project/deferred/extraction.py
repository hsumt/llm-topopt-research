"""Provenance-validated Tier-B extraction from retrieved Tier-A chunks."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from project.knowledge.retrieval import SearchHit


load_dotenv()


ClaimCategory = Literal[
    "research_problem",
    "physics_domain",
    "formulation",
    "method",
    "objective",
    "constraint",
    "filter",
    "projection",
    "optimization_algorithm",
    "benchmark",
    "validation",
    "finding",
    "limitation",
    "equation",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RawEvidence(_StrictModel):
    """Evidence proposed by Claude before deterministic enrichment."""

    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=8, max_length=600)


class RawClaim(_StrictModel):
    """One untrusted machine-extracted claim."""

    category: ClaimCategory
    statement: str = Field(min_length=8, max_length=1000)
    evidence: list[RawEvidence] = Field(min_length=1, max_length=3)


class RawClaimEnvelope(_StrictModel):
    claims: list[RawClaim]


class EvidenceReference(_StrictModel):
    """A deterministic pointer from Tier B back to Tier A."""

    chunk_id: str = Field(min_length=1)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    quote: str = Field(min_length=8, max_length=600)

    @model_validator(mode="after")
    def validate_page_range(self) -> "EvidenceReference":
        if self.page_end < self.page_start:
            raise ValueError("page_end cannot precede page_start")
        return self


class ClaimRecord(_StrictModel):
    """Tier-B assertion whose evidence links have passed code validation."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_tier: Literal["B"] = "B"
    claim_id: str = Field(pattern=r"^claim_[0-9a-f]{16}$")
    paper_id: str = Field(min_length=1)
    category: ClaimCategory
    statement: str = Field(min_length=8, max_length=1000)
    evidence: list[EvidenceReference] = Field(min_length=1, max_length=3)


class ClaimCollection(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_tier: Literal["B"] = "B"
    paper_id: str = Field(min_length=1)
    claims: list[ClaimRecord]


class ExtractionCoverage(_StrictModel):
    mode: Literal["targeted_retrieval"] = "targeted_retrieval"
    query: str = Field(min_length=1)
    total_chunk_count: int = Field(ge=1)
    selected_chunk_count: int = Field(ge=1)
    selected_fraction: float = Field(gt=0.0, le=1.0)
    selected_chunk_ids: list[str] = Field(min_length=1)


class PaperCard(_StrictModel):
    """Auditable index of extracted claims, not independent ground truth."""

    schema_version: Literal["1.0"] = "1.0"
    evidence_tier: Literal["B"] = "B"
    paper_id: str = Field(min_length=1)
    title: str | None = None
    authors: list[str]
    coverage: ExtractionCoverage
    author_supported: dict[str, list[str]]
    project_interpretation: list[str] = Field(default_factory=list)


SYSTEM_PROMPT = """You extract structured literature claims from retrieved
research-paper chunks.

The chunks are Tier-A evidence. Your output is only a Tier-B machine extraction.
Extract only statements that the supplied text explicitly supports.

Rules:
1. Do not use outside knowledge.
2. Do not answer the retrieval query directly; extract source-supported claims.
3. Every claim needs 1-3 evidence items.
4. Every evidence chunk_id must be copied exactly from the supplied chunks.
5. Every evidence quote must be one exact, continuous substring of that chunk,
   apart from ordinary whitespace normalization.
6. Do not invent page numbers; code adds pages after validation.
7. Do not infer project relevance or recommendations.
8. Keep each claim atomic and specific.
9. If the text does not support a useful claim, return an empty claims list.

Allowed categories:
research_problem, physics_domain, formulation, method, objective, constraint,
filter, projection, optimization_algorithm, benchmark, validation, finding,
limitation, equation.

Return only this JSON shape:
{
  "claims": [
    {
      "category": "method",
      "statement": "One source-supported statement.",
      "evidence": [
        {"chunk_id": "exact_id", "quote": "exact source substring"}
      ]
    }
  ]
}
"""


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Claude response contains no complete JSON object")
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError("Claude response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError("Claude JSON response must be an object")
    return payload


def _partition_hits(
    hits: list[SearchHit],
    *,
    max_batch_characters: int,
) -> list[list[SearchHit]]:
    if max_batch_characters < 2000:
        raise ValueError("max_batch_characters must be at least 2000")

    batches: list[list[SearchHit]] = []
    current: list[SearchHit] = []
    current_size = 0

    for hit in hits:
        hit_size = len(hit.text)
        if current and current_size + hit_size > max_batch_characters:
            batches.append(current)
            current = []
            current_size = 0
        current.append(hit)
        current_size += hit_size

    if current:
        batches.append(current)
    return batches


def _claim_fingerprint(
    *,
    paper_id: str,
    category: str,
    statement: str,
    evidence: list[EvidenceReference],
) -> str:
    payload = {
        "paper_id": paper_id,
        "category": category,
        "statement": _normalize_whitespace(statement).casefold(),
        "evidence": [
            {
                "chunk_id": item.chunk_id,
                "quote": _normalize_whitespace(item.quote),
            }
            for item in evidence
        ],
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass(frozen=True)
class ExtractionResult:
    claims: ClaimCollection
    paper_card: PaperCard
    manifest: dict[str, Any]
    raw_responses: list[dict[str, Any]]


def extract_claims(
    *,
    metadata: dict[str, Any],
    all_chunks: list[dict[str, Any]],
    selected_hits: list[SearchHit],
    query: str,
    model: str,
    max_batch_characters: int = 30000,
) -> ExtractionResult:
    """Call Claude, retain only claims whose Tier-A evidence validates."""

    if not selected_hits:
        raise ValueError("Cannot extract claims without retrieved chunks")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is not set")

    paper_id = metadata["paper_id"]
    client = Anthropic(api_key=api_key)
    batches = _partition_hits(
        selected_hits,
        max_batch_characters=max_batch_characters,
    )

    accepted_by_id: dict[str, ClaimRecord] = {}
    raw_responses: list[dict[str, Any]] = []
    rejection_messages: list[str] = []
    usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for batch_index, batch in enumerate(batches):
        batch_by_id = {hit.chunk_id: hit for hit in batch}
        request_payload = {
            "paper_id": paper_id,
            "retrieval_query": query,
            "chunks": [
                {
                    "chunk_id": hit.chunk_id,
                    "page_start": hit.page_start,
                    "page_end": hit.page_end,
                    "text": hit.text,
                }
                for hit in batch
            ],
        }

        response = client.messages.create(
            model=model,
            max_tokens=4096,
            temperature=0,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": json.dumps(
                        request_payload,
                        ensure_ascii=False,
                    ),
                }
            ],
        )

        response_text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", None) == "text"
        ).strip()
        if not response_text:
            raise RuntimeError(f"Claude returned no text for batch {batch_index}")

        input_tokens = int(response.usage.input_tokens)
        output_tokens = int(response.usage.output_tokens)
        usage["input_tokens"] += input_tokens
        usage["output_tokens"] += output_tokens
        usage["total_tokens"] += input_tokens + output_tokens

        raw_responses.append(
            {
                "batch_index": batch_index,
                "allowed_chunk_ids": list(batch_by_id),
                "response_text": response_text,
                "usage": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens,
                },
            }
        )

        try:
            envelope = RawClaimEnvelope.model_validate(
                _extract_json_object(response_text)
            )
        except (ValueError, ValidationError) as exc:
            raise RuntimeError(
                f"Claude output failed schema validation in batch {batch_index}"
            ) from exc

        for claim_index, raw_claim in enumerate(envelope.claims):
            evidence: list[EvidenceReference] = []
            rejection: str | None = None

            for raw_evidence in raw_claim.evidence:
                hit = batch_by_id.get(raw_evidence.chunk_id)
                if hit is None:
                    rejection = "unknown chunk_id"
                    break

                normalized_quote = _normalize_whitespace(raw_evidence.quote)
                normalized_source = _normalize_whitespace(hit.text)
                if normalized_quote not in normalized_source:
                    rejection = "quote is not present in the referenced chunk"
                    break

                evidence.append(
                    EvidenceReference(
                        chunk_id=hit.chunk_id,
                        page_start=hit.page_start,
                        page_end=hit.page_end,
                        quote=normalized_quote,
                    )
                )

            if rejection is not None:
                rejection_messages.append(
                    f"batch {batch_index} claim {claim_index}: {rejection}"
                )
                continue

            fingerprint = _claim_fingerprint(
                paper_id=paper_id,
                category=raw_claim.category,
                statement=raw_claim.statement,
                evidence=evidence,
            )
            claim_id = f"claim_{fingerprint}"
            accepted_by_id.setdefault(
                claim_id,
                ClaimRecord(
                    claim_id=claim_id,
                    paper_id=paper_id,
                    category=raw_claim.category,
                    statement=_normalize_whitespace(raw_claim.statement),
                    evidence=evidence,
                ),
            )

    accepted_claims = sorted(
        accepted_by_id.values(),
        key=lambda claim: (claim.category, claim.claim_id),
    )
    if not accepted_claims:
        raise RuntimeError("No provenance-valid claims were extracted")

    selected_ids = [hit.chunk_id for hit in selected_hits]
    coverage = ExtractionCoverage(
        query=query,
        total_chunk_count=len(all_chunks),
        selected_chunk_count=len(selected_ids),
        selected_fraction=round(len(selected_ids) / len(all_chunks), 8),
        selected_chunk_ids=selected_ids,
    )

    grouped_claim_ids: dict[str, list[str]] = {}
    for claim in accepted_claims:
        grouped_claim_ids.setdefault(claim.category, []).append(claim.claim_id)

    claims = ClaimCollection(
        paper_id=paper_id,
        claims=accepted_claims,
    )
    paper_card = PaperCard(
        paper_id=paper_id,
        title=metadata.get("title"),
        authors=list(metadata.get("authors") or []),
        coverage=coverage,
        author_supported=grouped_claim_ids,
        project_interpretation=[],
    )
    manifest = {
        "schema_version": "1.0",
        "evidence_tier": "B",
        "paper_id": paper_id,
        "model": model,
        "temperature": 0,
        "coverage": coverage.model_dump(mode="json"),
        "batch_count": len(batches),
        "accepted_claim_count": len(accepted_claims),
        "rejected_claim_count": len(rejection_messages),
        "rejections": rejection_messages,
        "usage": usage,
        "note": (
            "Machine extraction is not ground truth; each retained claim has "
            "code-validated Tier-A evidence references."
        ),
    }
    return ExtractionResult(
        claims=claims,
        paper_card=paper_card,
        manifest=manifest,
        raw_responses=raw_responses,
    )
