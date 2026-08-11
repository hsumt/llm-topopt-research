"""Build immutable, provenance-checked literature context for the critic."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from project.knowledge.approval import APPROVAL_FILENAME, APPROVAL_SCOPE
from project.knowledge.extraction import (
    ClaimCollection,
    ClaimRecord,
    EvidenceReference,
    _normalize_whitespace,
)
from project.knowledge.retrieval import BM25Index, load_paper_artifacts


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON: {path}") from exc


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ApprovedClaim(_StrictModel):
    claim_id: str = Field(pattern=r"^claim_[0-9a-f]{16}$")
    paper_id: str = Field(min_length=1)
    title: str | None = None
    authors: list[str]
    category: str = Field(min_length=1)
    statement: str = Field(min_length=8)
    evidence: list[EvidenceReference] = Field(min_length=1)


class CriticEvidencePacket(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    evidence_tier: Literal["A+B"] = "A+B"
    role: Literal["critic_interpretation_only"] = (
        "critic_interpretation_only"
    )
    validation_authority: Literal["deterministic_python_only"] = (
        "deterministic_python_only"
    )
    may_modify_solver_or_spec: Literal[False] = False
    query: str = Field(min_length=1)
    retrieval_method: Literal["bm25_over_approved_claims_v1"] = (
        "bm25_over_approved_claims_v1"
    )
    eligible_claim_count: int = Field(ge=1)
    selected_claim_count: int = Field(ge=1)
    selected_claim_ids: list[str] = Field(min_length=1)
    claims: list[ApprovedClaim] = Field(min_length=1)
    frozen_artifacts: dict[str, str]
    approval: dict[str, Any]

    @model_validator(mode="after")
    def check_selected_ids(self) -> "CriticEvidencePacket":
        claim_ids = [claim.claim_id for claim in self.claims]
        if claim_ids != self.selected_claim_ids:
            raise ValueError("selected_claim_ids must match claims in order")
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("selected claims contain duplicate IDs")
        if self.selected_claim_count != len(claim_ids):
            raise ValueError("selected_claim_count is inconsistent")
        if self.selected_claim_count > self.eligible_claim_count:
            raise ValueError("selected count exceeds eligible count")
        return self


def build_run_query(critic_input: dict[str, Any]) -> str:
    """Create a stable retrieval query from fields the critic already receives."""

    run_config = critic_input.get("run_config") or {}
    final_result = critic_input.get("final_result") or {}
    validation = critic_input.get("validation") or {}

    terms = [
        "topology optimization",
        "SIMP",
        "compliance objective",
        "volume constraint",
        "density sensitivity",
        str(critic_input.get("problem") or ""),
        str(run_config.get("kinematics") or ""),
        str(run_config.get("filter") or ""),
        str(run_config.get("projection") or ""),
        "Heaviside continuation" if run_config.get("beta_schedule_requested") else "",
        "design convergence" if final_result.get("design_converged") else "",
        "objective plateau" if final_result.get("objective_plateau") else "",
    ]

    for name, check in sorted((validation.get("checks") or {}).items()):
        if not isinstance(check, dict):
            continue
        if check.get("passed") is not True:
            terms.append(name.replace("_", " "))
    for warning in validation.get("quality_warnings") or []:
        terms.append(str(warning))

    normalized = [re.sub(r"\s+", " ", term).strip() for term in terms]
    return " ".join(term for term in normalized if term)


def _validate_claim(
    claim: ClaimRecord,
    *,
    paper_id: str,
    chunks_by_id: dict[str, dict[str, Any]],
) -> None:
    if claim.paper_id != paper_id:
        raise RuntimeError(f"Claim {claim.claim_id} has the wrong paper_id")
    for evidence in claim.evidence:
        chunk = chunks_by_id.get(evidence.chunk_id)
        if chunk is None:
            raise RuntimeError(
                f"Claim {claim.claim_id} references unknown chunk "
                f"{evidence.chunk_id}"
            )
        if (
            evidence.page_start != chunk["page_start"]
            or evidence.page_end != chunk["page_end"]
        ):
            raise RuntimeError(
                f"Claim {claim.claim_id} has incorrect page provenance"
            )
        if _normalize_whitespace(evidence.quote) not in _normalize_whitespace(
            chunk["text"]
        ):
            raise RuntimeError(
                f"Claim {claim.claim_id} contains an unsupported quote"
            )


def _load_approved_claims(
    paper_directory: Path,
    approval_path: Path,
) -> tuple[dict[str, Any], list[ClaimRecord], dict[str, str], dict[str, Any]]:
    metadata, chunks, chunks_sha256 = load_paper_artifacts(paper_directory)
    claims_path = paper_directory / "claims.json"
    claims = ClaimCollection.model_validate(_read_json(claims_path))
    approval = _read_json(approval_path)

    if approval.get("schema_version") != "1.0":
        raise RuntimeError("Unsupported approval schema version")
    if approval.get("approval_scope") != APPROVAL_SCOPE:
        raise RuntimeError("Approval is not scoped to critic interpretation")
    if approval.get("paper_id") != metadata["paper_id"]:
        raise RuntimeError("Approval paper_id does not match metadata")
    if approval.get("source_sha256") != metadata["source_sha256"]:
        raise RuntimeError("Approval source hash is stale")
    if approval.get("chunks_sha256") != chunks_sha256:
        raise RuntimeError("Approval chunks hash is stale")
    if approval.get("claims_sha256") != sha256_file(claims_path):
        raise RuntimeError("Approval claims hash is stale")

    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}
    claims_by_id = {claim.claim_id: claim for claim in claims.claims}
    approved_ids = approval.get("approved_claim_ids")
    if not isinstance(approved_ids, list) or not approved_ids:
        raise RuntimeError("Approval contains no approved claim IDs")
    if len(approved_ids) != len(set(approved_ids)):
        raise RuntimeError("Approval contains duplicate claim IDs")

    approved_claims: list[ClaimRecord] = []
    for claim_id in approved_ids:
        claim = claims_by_id.get(claim_id)
        if claim is None:
            raise RuntimeError(f"Approval references unknown claim {claim_id}")
        _validate_claim(
            claim,
            paper_id=metadata["paper_id"],
            chunks_by_id=chunks_by_id,
        )
        approved_claims.append(claim)

    frozen = {
        "metadata_sha256": sha256_file(paper_directory / "metadata.json"),
        "chunks_sha256": chunks_sha256,
        "claims_sha256": sha256_file(claims_path),
        "paper_card_sha256": sha256_file(paper_directory / "paper_card.json"),
        "extraction_manifest_sha256": sha256_file(
            paper_directory / "extraction_manifest.json"
        ),
        "approval_sha256": sha256_file(approval_path),
    }
    return metadata, approved_claims, frozen, approval


def build_critic_evidence(
    *,
    critic_input: dict[str, Any],
    paper_directory: str | Path,
    approval_path: str | Path | None = None,
    top_k: int = 6,
) -> CriticEvidencePacket:
    """Return only reviewed claims whose Tier-A links still validate."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    root = Path(paper_directory)
    resolved_approval = (
        Path(approval_path) if approval_path else root / APPROVAL_FILENAME
    )
    metadata, eligible_claims, frozen, approval = _load_approved_claims(
        root,
        resolved_approval,
    )
    query = build_run_query(critic_input)

    pseudo_chunks = []
    claims_by_id: dict[str, ClaimRecord] = {}
    for index, claim in enumerate(eligible_claims):
        claims_by_id[claim.claim_id] = claim
        pages = [item.page_start for item in claim.evidence] + [
            item.page_end for item in claim.evidence
        ]
        pseudo_chunks.append(
            {
                "paper_id": metadata["paper_id"],
                "chunk_id": claim.claim_id,
                "chunk_index": index,
                "page_start": min(pages),
                "page_end": max(pages),
                "text": " ".join(
                    [claim.category, claim.statement]
                    + [item.quote for item in claim.evidence]
                ),
            }
        )

    hits = BM25Index(pseudo_chunks).search(
        query,
        top_k=min(top_k, len(pseudo_chunks)),
    )
    if not hits:
        raise RuntimeError("No approved claims matched the run-derived query")

    selected = [claims_by_id[hit.chunk_id] for hit in hits]
    approved_claims = [
        ApprovedClaim(
            claim_id=claim.claim_id,
            paper_id=claim.paper_id,
            title=metadata.get("title"),
            authors=list(metadata.get("authors") or []),
            category=claim.category,
            statement=claim.statement,
            evidence=claim.evidence,
        )
        for claim in selected
    ]
    selected_ids = [claim.claim_id for claim in approved_claims]

    return CriticEvidencePacket(
        query=query,
        eligible_claim_count=len(eligible_claims),
        selected_claim_count=len(approved_claims),
        selected_claim_ids=selected_ids,
        claims=approved_claims,
        frozen_artifacts=frozen,
        approval={
            "scope": approval["approval_scope"],
            "review": approval["review"],
            "approved_claim_ids": approval["approved_claim_ids"],
        },
    )
