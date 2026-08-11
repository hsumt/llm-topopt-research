"""Run an evidence-aware critic without changing solver or validation code."""

from __future__ import annotations

import argparse
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from project.knowledge.critic_evidence import (
    CriticEvidencePacket,
    build_critic_evidence,
    canonical_sha256,
    sha256_file,
)


load_dotenv()


SYSTEM_PROMPT = """You are an evidence-limited structural topology
optimization critic. You receive (1) a compact deterministic run packet and
(2) reviewed literature claims with exact source quotations and page ranges.

The deterministic Python validation object is the only authority over whether
the run passed. Literature is interpretation-only evidence.

Tasks:
1. Summarize convergence behavior using exact run metrics.
2. Flag engineering anomalies using only provided deterministic metrics.
3. State a cautious physical-plausibility interpretation.
4. Explain any relevant literature context using only supplied claim records.
5. Give one specific recommendation for a future run.

Rules:
- Never issue an independent pass/fail verdict.
- Never modify or propose modifications to the current run, solver equations,
  FEM code, ProblemSpec, or validation result.
- Do not invent metrics, image observations, equations, claims, claim IDs, page
  numbers, or causal relationships.
- No topology image content is supplied. Filenames are not visual evidence.
- Do not claim visible connectivity, checkerboard absence, symmetry, member
  quality, or support/load consistency without a deterministic metric.
- Treat objective plateau, design convergence, and continuation completion as
  separate statuses.
- Do not use objective plateau alone as proof of optimizer convergence.
- If semantic assurance is unconfirmed, conclusions apply only to the parsed
  model.
- Do not recommend a causal parameter change without an A/B comparison or an
  explicit deterministic rule in the packet.
- A provenance-valid claim is still a source statement, not proof that the
  current run exhibits the same behavior.
- In literature_context, cite only supplied claim IDs. Do not output pages;
  Python derives pages from the validated evidence.

Return only one JSON object with exactly this shape:
{
  "convergence_behavior": "2-3 sentences",
  "engineering_anomalies": ["zero or more observations"],
  "physical_plausibility": "cautious interpretation",
  "literature_context": [
    {
      "interpretation": "what the source says and how narrowly it informs interpretation",
      "claim_ids": ["claim_0123456789abcdef"]
    }
  ],
  "recommendation": "one future-run recommendation",
  "limitations": ["important scope limitations"]
}
"""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class LiteratureInterpretation(_StrictModel):
    interpretation: str = Field(min_length=8, max_length=1600)
    claim_ids: list[str] = Field(min_length=1, max_length=4)


class EvidenceCriticResponse(_StrictModel):
    convergence_behavior: str = Field(min_length=8, max_length=2400)
    engineering_anomalies: list[str]
    physical_plausibility: str = Field(min_length=8, max_length=2400)
    literature_context: list[LiteratureInterpretation]
    recommendation: str = Field(min_length=8, max_length=1600)
    limitations: list[str]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Required run artifact is missing: {path}")
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


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start < 0 or end < start:
        raise RuntimeError("Critic response contains no complete JSON object")
    try:
        payload = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("Critic response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Critic JSON response must be an object")
    return payload


def _citation_label(claim: Any) -> str:
    authors = claim.authors
    if not authors:
        author_label = claim.paper_id
    elif len(authors) == 1:
        author_label = authors[0]
    elif len(authors) == 2:
        author_label = f"{authors[0]} and {authors[1]}"
    else:
        author_label = f"{authors[0]} et al."

    pages = sorted(
        {
            (evidence.page_start, evidence.page_end)
            for evidence in claim.evidence
        }
    )
    page_label = ", ".join(
        str(start) if start == end else f"{start}-{end}"
        for start, end in pages
    )
    return f"{author_label}, pp. {page_label}; {claim.claim_id}"


def validate_response_citations(
    response: EvidenceCriticResponse,
    evidence: CriticEvidencePacket,
) -> None:
    allowed = set(evidence.selected_claim_ids)
    used: set[str] = set()
    for item in response.literature_context:
        if len(item.claim_ids) != len(set(item.claim_ids)):
            raise RuntimeError("Critic repeated a claim ID in one citation")
        unknown = set(item.claim_ids) - allowed
        if unknown:
            raise RuntimeError(
                f"Critic cited claims outside its evidence packet: {sorted(unknown)}"
            )
        used.update(item.claim_ids)
    if response.literature_context and not used:
        raise RuntimeError("Literature interpretation has no valid citations")


def render_summary(
    response: EvidenceCriticResponse,
    evidence: CriticEvidencePacket,
) -> str:
    claims_by_id = {claim.claim_id: claim for claim in evidence.claims}
    lines = [
        "Convergence Behavior",
        response.convergence_behavior,
        "",
        "Engineering Anomalies",
    ]
    lines.extend(
        f"- {item}" for item in response.engineering_anomalies
    )
    if not response.engineering_anomalies:
        lines.append("- None supported by the supplied deterministic metrics.")

    lines.extend(
        [
            "",
            "Physical Plausibility",
            response.physical_plausibility,
            "",
            "Literature Context",
        ]
    )
    for item in response.literature_context:
        citations = "; ".join(
            _citation_label(claims_by_id[claim_id])
            for claim_id in item.claim_ids
        )
        lines.append(f"- {item.interpretation} [{citations}]")
    if not response.literature_context:
        lines.append("- No supplied claim was needed for this interpretation.")

    lines.extend(
        [
            "",
            "Recommendation",
            response.recommendation,
            "",
            "Limitations",
        ]
    )
    lines.extend(f"- {item}" for item in response.limitations)
    if not response.limitations:
        lines.append("- No additional limitation stated.")
    return "\n".join(lines).rstrip() + "\n"


def _response_text(response: Any) -> str:
    return "".join(
        block.text
        for block in response.content
        if getattr(block, "type", None) == "text"
    ).strip()


def run_evidence_critic(
    *,
    run_directory: str | Path,
    paper_directory: str | Path,
    approval_path: str | Path | None = None,
    top_k: int = 6,
    model: str = "claude-sonnet-4-6",
    client: Any | None = None,
) -> dict[str, Any]:
    """Create a supplemental critic result; never mutate the base run packet."""

    run_root = Path(run_directory)
    critic_input_path = run_root / "critic_input.json"
    result_packet_path = run_root / "result_packet.json"
    critic_input = _read_json(critic_input_path)
    original_packet = _read_json(result_packet_path)

    input_validation = critic_input.get("validation")
    packet_validation = original_packet.get("validation")
    if input_validation != packet_validation:
        raise RuntimeError(
            "Validation differs between critic_input.json and result_packet.json"
        )
    if not isinstance(input_validation, dict):
        raise RuntimeError("Run packet has no deterministic validation object")
    if input_validation.get("passed") is not True:
        manifest = {
            "schema_version": "1.0",
            "status": "skipped_hard_validation_failed",
            "validation_authority": "deterministic_python_only",
            "critic_input_sha256": sha256_file(critic_input_path),
            "result_packet_sha256": sha256_file(result_packet_path),
            "api_called": False,
        }
        _write_json(run_root / "literature_critic_manifest.json", manifest)
        return manifest

    evidence = build_critic_evidence(
        critic_input=critic_input,
        paper_directory=paper_directory,
        approval_path=approval_path,
        top_k=top_k,
    )
    augmented_input = deepcopy(critic_input)
    augmented_input["literature_evidence"] = evidence.model_dump(mode="json")
    augmented_input["literature_evidence_scope"] = {
        "evidence_is_interpretation_only": True,
        "validation_authority": "deterministic_python_only",
        "may_modify_current_run": False,
        "topology_image_content_included": False,
    }

    api_client = client
    if api_client is None:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        api_client = Anthropic(api_key=api_key)

    response = api_client.messages.create(
        model=model,
        max_tokens=1800,
        temperature=0,
        system=SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": json.dumps(
                    augmented_input,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            }
        ],
    )
    raw_text = _response_text(response)
    if not raw_text:
        raise RuntimeError("Evidence critic returned no text")
    try:
        validated = EvidenceCriticResponse.model_validate(
            _extract_json_object(raw_text)
        )
    except ValidationError as exc:
        raise RuntimeError("Evidence critic failed response schema") from exc
    validate_response_citations(validated, evidence)
    summary = render_summary(validated, evidence)

    input_tokens = int(response.usage.input_tokens)
    output_tokens = int(response.usage.output_tokens)
    manifest = {
        "schema_version": "1.0",
        "status": "completed",
        "validation_authority": "deterministic_python_only",
        "literature_authority": "critic_interpretation_only",
        "model": model,
        "temperature": 0,
        "critic_input_sha256": sha256_file(critic_input_path),
        "result_packet_sha256": sha256_file(result_packet_path),
        "evidence_packet_sha256": canonical_sha256(
            evidence.model_dump(mode="json")
        ),
        "selected_claim_ids": evidence.selected_claim_ids,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        },
        "artifacts": [
            "literature_evidence.json",
            "literature_critic_input.json",
            "literature_critic_raw.json",
            "literature_critic_response.json",
            "literature_critic_summary.txt",
            "literature_critic_manifest.json",
            "result_packet_with_literature.json",
        ],
    }

    _write_json(
        run_root / "literature_evidence.json",
        evidence.model_dump(mode="json"),
    )
    _write_json(run_root / "literature_critic_input.json", augmented_input)
    _write_json(
        run_root / "literature_critic_raw.json",
        {"schema_version": "1.0", "response_text": raw_text},
    )
    _write_json(
        run_root / "literature_critic_response.json",
        validated.model_dump(mode="json"),
    )
    (run_root / "literature_critic_summary.txt").write_text(
        summary,
        encoding="utf-8",
    )
    _write_json(run_root / "literature_critic_manifest.json", manifest)

    augmented_packet = deepcopy(original_packet)
    augmented_packet["literature_critic"] = {
        "status": "completed",
        "summary": summary,
        "selected_claim_ids": evidence.selected_claim_ids,
        "evidence_artifact": "literature_evidence.json",
        "manifest_artifact": "literature_critic_manifest.json",
    }
    if augmented_packet.get("validation") != original_packet.get("validation"):
        raise RuntimeError("K5 attempted to change deterministic validation")
    _write_json(
        run_root / "result_packet_with_literature.json",
        augmented_packet,
    )
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a supplemental evidence-aware critic on an existing run."
        )
    )
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--paper-directory", type=Path, required=True)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument(
        "--model",
        default=os.getenv("KNOWLEDGE_CRITIC_MODEL", "claude-sonnet-4-6"),
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    manifest = run_evidence_critic(
        run_directory=args.run_directory,
        paper_directory=args.paper_directory,
        approval_path=args.approval,
        top_k=args.top_k,
        model=args.model,
    )
    if manifest["status"] == "skipped_hard_validation_failed":
        print("K5 LITERATURE CRITIC: SKIPPED")
        print("reason: deterministic hard validation failed")
        print("API called: no")
        return

    print("K5 APPROVAL + PROVENANCE GATE: PASS")
    print("selected claims:", len(manifest["selected_claim_ids"]))
    print("K5 EVIDENCE-LIMITED CRITIC: PASS")
    print("validation authority: deterministic Python only")
    print("output directory:", args.run_directory)


if __name__ == "__main__":
    main()
