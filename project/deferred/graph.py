"""Deterministic JSON graph derived from validated Tier-B claim records."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from project.knowledge.extraction import ClaimCollection, EvidenceReference


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class GraphNode(_StrictModel):
    node_id: str = Field(min_length=1)
    node_type: str = Field(min_length=1)
    label: str = Field(min_length=1)
    evidence_tier: Literal["A", "B"]
    evidence: list[EvidenceReference]


class GraphEdge(_StrictModel):
    edge_id: str = Field(pattern=r"^edge_[0-9a-f]{16}$")
    source: str = Field(min_length=1)
    relation: str = Field(min_length=1)
    target: str = Field(min_length=1)
    evidence_tier: Literal["B"] = "B"
    evidence: list[EvidenceReference] = Field(min_length=1)


class KnowledgeGraph(_StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    graph_type: Literal["provenance_aware_literature_v1"] = (
        "provenance_aware_literature_v1"
    )
    nodes: list[GraphNode]
    edges: list[GraphEdge]


_NODE_TYPES = {
    "research_problem": "ResearchProblem",
    "physics_domain": "PhysicsDomain",
    "formulation": "Formulation",
    "method": "Method",
    "objective": "Objective",
    "constraint": "Constraint",
    "filter": "Filter",
    "projection": "Projection",
    "optimization_algorithm": "OptimizationAlgorithm",
    "benchmark": "Benchmark",
    "validation": "ValidationMethod",
    "finding": "Finding",
    "limitation": "Limitation",
    "equation": "Equation",
}

_RELATIONS = {
    "formulation": "USES_METHOD",
    "method": "USES_METHOD",
    "filter": "USES_METHOD",
    "projection": "USES_METHOD",
    "optimization_algorithm": "USES_METHOD",
    "objective": "USES_OBJECTIVE",
    "constraint": "USES_CONSTRAINT",
    "benchmark": "EVALUATES_ON",
    "equation": "DEFINES",
}


def _edge_id(source: str, relation: str, target: str) -> str:
    payload = f"{source}\0{relation}\0{target}".encode("utf-8")
    return "edge_" + hashlib.sha256(payload).hexdigest()[:16]


def build_graph(
    metadata: dict[str, Any],
    claims: ClaimCollection,
) -> KnowledgeGraph:
    """Map validated claims to typed nodes and evidence-backed edges."""

    paper_id = claims.paper_id
    paper_node_id = f"paper:{paper_id}"
    nodes = [
        GraphNode(
            node_id=paper_node_id,
            node_type="Paper",
            label=metadata.get("title") or paper_id,
            evidence_tier="A",
            evidence=[],
        )
    ]
    edges: list[GraphEdge] = []

    for claim in claims.claims:
        claim_node_id = f"claim:{claim.claim_id}"
        nodes.append(
            GraphNode(
                node_id=claim_node_id,
                node_type=_NODE_TYPES[claim.category],
                label=claim.statement,
                evidence_tier="B",
                evidence=claim.evidence,
            )
        )
        relation = _RELATIONS.get(claim.category, "DISCUSSES")
        edges.append(
            GraphEdge(
                edge_id=_edge_id(paper_node_id, relation, claim_node_id),
                source=paper_node_id,
                relation=relation,
                target=claim_node_id,
                evidence=claim.evidence,
            )
        )

    nodes.sort(key=lambda node: node.node_id)
    edges.sort(key=lambda edge: edge.edge_id)
    return KnowledgeGraph(nodes=nodes, edges=edges)
