"""Context ingestion and lightweight deterministic retrieval.

The pre-solve system treats user-supplied context as evidence, not intent. This
module extracts searchable text from plain-text/PDF assets when possible and
retrieves small relevant snippets before the LLM asks the engineer a question.
Images remain available as visual evidence to the multimodal LLM and UI.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
from dataclasses import dataclass
from typing import Iterable

from project.formulation.models import AttachmentRef, ContextSnippet


@dataclass(frozen=True)
class ContextAttachment:
    name: str
    media_type: str
    data: bytes


_TOKEN_RE = re.compile(r"[A-Za-z0-9_+./-]+")


def attachment_id(name: str, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()[:12]
    safe = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")[:32] or "asset"
    return f"asset_{safe}_{digest}"


def attachment_ref(attachment: ContextAttachment) -> AttachmentRef:
    media = attachment.media_type.lower()
    if media.startswith("image/"):
        kind = "image"
    elif media == "application/pdf":
        kind = "pdf"
    elif media.startswith("text/") or attachment.name.lower().endswith(
        (".txt", ".md", ".csv", ".json")
    ):
        kind = "text"
    else:
        kind = "other"
    return AttachmentRef(
        id=attachment_id(attachment.name, attachment.data),
        name=attachment.name,
        media_type=attachment.media_type,
        size_bytes=len(attachment.data),
        kind=kind,
    )


def _extract_pdf_text(data: bytes) -> str:
    try:
        from pypdf import PdfReader
    except Exception:
        return ""
    try:
        reader = PdfReader(io.BytesIO(data))
        return "\n\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception:
        return ""


def extract_attachment_text(attachment: ContextAttachment) -> str:
    media = attachment.media_type.lower()
    name = attachment.name.lower()
    if media == "application/pdf" or name.endswith(".pdf"):
        return _extract_pdf_text(attachment.data)
    if media.startswith("text/") or name.endswith((".txt", ".md", ".csv", ".json")):
        try:
            return attachment.data.decode("utf-8")
        except UnicodeDecodeError:
            return attachment.data.decode("utf-8", errors="replace")
    return ""


def _chunk_text(text: str, *, max_chars: int = 1200) -> list[str]:
    text = re.sub(r"\r\n?", "\n", text).strip()
    if not text:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        if len(paragraph) > max_chars:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        else:
            sentences = [paragraph]
        for piece in sentences:
            piece = piece.strip()
            if not piece:
                continue
            candidate = f"{current}\n{piece}".strip() if current else piece
            if len(candidate) <= max_chars:
                current = candidate
            else:
                if current:
                    chunks.append(current)
                current = piece[:max_chars]
    if current:
        chunks.append(current)
    return chunks


def build_context_snippets(
    context: str | None,
    attachments: Iterable[ContextAttachment] | None = None,
) -> list[ContextSnippet]:
    snippets: list[ContextSnippet] = []
    index = 0
    if context and context.strip():
        for chunk in _chunk_text(context):
            index += 1
            snippets.append(
                ContextSnippet(
                    id=f"context_{index:03d}",
                    source="typed context",
                    text=chunk,
                )
            )
    for attachment in attachments or []:
        text = extract_attachment_text(attachment)
        for chunk in _chunk_text(text):
            index += 1
            snippets.append(
                ContextSnippet(
                    id=f"context_{index:03d}",
                    source=attachment.name,
                    text=chunk,
                )
            )
    return snippets


def _terms(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN_RE.findall(text) if len(token) >= 2]


def retrieve_context(
    query: str,
    snippets: list[ContextSnippet],
    *,
    limit: int = 8,
) -> list[ContextSnippet]:
    """Small BM25-like lexical retriever with no external index dependency."""

    if not snippets:
        return []
    q_terms = _terms(query)
    if not q_terms:
        return snippets[:limit]

    doc_terms = [_terms(item.text) for item in snippets]
    n_docs = len(doc_terms)
    df: dict[str, int] = {}
    for terms in doc_terms:
        for term in set(terms):
            df[term] = df.get(term, 0) + 1

    scored: list[tuple[float, int]] = []
    q_set = set(q_terms)
    for i, terms in enumerate(doc_terms):
        if not terms:
            continue
        tf: dict[str, int] = {}
        for term in terms:
            tf[term] = tf.get(term, 0) + 1
        score = 0.0
        for term in q_set:
            if term not in tf:
                continue
            idf = math.log((n_docs + 1) / (df.get(term, 0) + 0.5)) + 1.0
            score += (1.0 + math.log(tf[term])) * idf
        if score > 0:
            scored.append((score, i))

    if not scored:
        return snippets[: min(limit, len(snippets))]
    scored.sort(reverse=True)
    result: list[ContextSnippet] = []
    for score, i in scored[:limit]:
        result.append(snippets[i].model_copy(update={"score": round(score, 6)}))
    return result


def retrieval_query_from_state(problem: str, parser_result) -> str:
    parts = [problem]
    for item in parser_result.unresolved_items:
        parts.extend([item.issue, item.question])
    for item in parser_result.contradictions:
        parts.append(item.description)
    return "\n".join(parts)


def render_pdf_pages(
    attachment: ContextAttachment,
    *,
    max_pages: int = 3,
    dpi: int = 96,
) -> list[tuple[str, bytes]]:
    """Render a few PDF pages to PNG for the engineer-facing UI only.

    This is optional and local.  It does not perform OCR or send another model
    call.  If PyMuPDF is unavailable, callers simply receive an empty list and
    can fall back to the browser PDF preview.
    """

    if attachment.media_type.lower() != "application/pdf" and not attachment.name.lower().endswith(".pdf"):
        return []
    try:
        import fitz  # PyMuPDF
    except Exception:
        return []
    try:
        doc = fitz.open(stream=attachment.data, filetype="pdf")
        scale = float(dpi) / 72.0
        matrix = fitz.Matrix(scale, scale)
        output: list[tuple[str, bytes]] = []
        for index in range(min(len(doc), max_pages)):
            page = doc.load_page(index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            output.append((f"{attachment.name} — page {index + 1}", pix.tobytes("png")))
        doc.close()
        return output
    except Exception:
        return []
