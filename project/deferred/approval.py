"""Create a human-review approval manifest for Tier-B literature claims.

The command never decides that a claim is correct.  It records an explicit
review decision and binds that decision to the exact chunks.json and
claims.json bytes that were reviewed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Any

from project.knowledge.extraction import ClaimCollection
from project.knowledge.retrieval import load_paper_artifacts


APPROVAL_FILENAME = "critic_approval.json"
APPROVAL_SCOPE = "critic_interpretation_only"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required artifact is missing: {path}")
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON: {path}") from exc


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def load_claims(paper_directory: str | Path) -> ClaimCollection:
    root = Path(paper_directory)
    return ClaimCollection.model_validate(_read_json(root / "claims.json"))


def render_claims_for_review(
    paper_directory: str | Path,
) -> str:
    root = Path(paper_directory)
    metadata, chunks, _ = load_paper_artifacts(root)
    claims = load_claims(root)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    lines = [
        f"Paper: {metadata.get('title') or metadata['paper_id']}",
        f"Paper ID: {metadata['paper_id']}",
        f"Claims: {len(claims.claims)}",
        "",
    ]
    for index, claim in enumerate(claims.claims, start=1):
        lines.append(
            f"{index}. {claim.claim_id} [{claim.category}]\n"
            f"   {claim.statement}"
        )
        for evidence in claim.evidence:
            source = chunks_by_id.get(evidence.chunk_id)
            resolved = source is not None
            lines.append(
                f"   - pp. {evidence.page_start}-{evidence.page_end}; "
                f"{evidence.chunk_id}; resolves={resolved}\n"
                f"     QUOTE: {evidence.quote}"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def create_approval_manifest(
    paper_directory: str | Path,
    *,
    reviewer: str,
    review_date: str,
    approved_claim_ids: list[str],
    output_path: str | Path | None = None,
) -> Path:
    """Record a review decision without changing source knowledge artifacts."""

    root = Path(paper_directory)
    metadata, _, chunks_sha256 = load_paper_artifacts(root)
    claims = load_claims(root)

    if not reviewer.strip():
        raise ValueError("reviewer cannot be blank")
    try:
        date.fromisoformat(review_date)
    except ValueError as exc:
        raise ValueError("review_date must use YYYY-MM-DD") from exc

    available_ids = {claim.claim_id for claim in claims.claims}
    requested_ids = set(approved_claim_ids)
    unknown = sorted(requested_ids - available_ids)
    if unknown:
        raise ValueError(f"Unknown claim IDs requested for approval: {unknown}")
    if not requested_ids:
        raise ValueError("At least one claim must be explicitly approved")

    destination = Path(output_path) if output_path else root / APPROVAL_FILENAME
    payload = {
        "schema_version": "1.0",
        "approval_scope": APPROVAL_SCOPE,
        "paper_id": metadata["paper_id"],
        "source_sha256": metadata["source_sha256"],
        "chunks_sha256": chunks_sha256,
        "claims_sha256": _sha256(root / "claims.json"),
        "approved_claim_ids": sorted(requested_ids),
        "review": {
            "reviewer": reviewer.strip(),
            "review_date": review_date,
            "statement": (
                "The listed Tier-B claims were reviewed for use as "
                "interpretive context only. This approval grants no "
                "solver, parameter-setting, or validation authority."
            ),
        },
    }
    _write_json(destination, payload)
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect claims or record explicit approval for critic-only use."
        )
    )
    parser.add_argument("paper_directory", type=Path)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print claims and evidence without writing an approval.",
    )
    parser.add_argument("--reviewer")
    parser.add_argument("--review-date")
    parser.add_argument(
        "--claim-id",
        action="append",
        default=[],
        help="Approved claim ID; repeat for multiple claims.",
    )
    parser.add_argument(
        "--approve-all",
        action="store_true",
        help="Approve all displayed claims after review.",
    )
    parser.add_argument("--output", type=Path)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    review_text = render_claims_for_review(args.paper_directory)
    print(review_text, end="")

    if args.list:
        print("REVIEW ONLY: no approval file written")
        return

    if not args.reviewer or not args.review_date:
        raise SystemExit(
            "--reviewer and --review-date are required when recording approval"
        )
    if args.approve_all and args.claim_id:
        raise SystemExit("Use either --approve-all or --claim-id, not both")

    claims = load_claims(args.paper_directory)
    approved_ids = (
        [claim.claim_id for claim in claims.claims]
        if args.approve_all
        else args.claim_id
    )
    output = create_approval_manifest(
        args.paper_directory,
        reviewer=args.reviewer,
        review_date=args.review_date,
        approved_claim_ids=approved_ids,
        output_path=args.output,
    )
    print("K5 CLAIM APPROVAL MANIFEST: PASS")
    print("approved claims:", len(approved_ids))
    print("output:", output)


if __name__ == "__main__":
    main()
