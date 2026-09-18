"""Build Anthropic multimodal user-message content blocks."""

from __future__ import annotations

import base64
import json

from project.formulation.context import ContextAttachment


_SUPPORTED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/webp", "image/gif"}


def build_user_content(
    payload: dict,
    attachments: list[ContextAttachment] | None = None,
    *,
    max_visual_assets: int = 4,
    max_asset_bytes: int = 8_000_000,
):
    """Return Anthropic content blocks with JSON first and visual evidence after.

    Text/PDF extraction is handled separately by the context retriever. Images
    and PDFs are also attached directly so the model can inspect diagrams,
    sketches and layout when the provider/model supports those content types.
    """

    blocks: list[dict] = [
        {
            "type": "text",
            "text": json.dumps(payload, separators=(",", ":"), ensure_ascii=False),
        }
    ]
    added = 0
    for attachment in attachments or []:
        if added >= max_visual_assets or len(attachment.data) > max_asset_bytes:
            continue
        media = attachment.media_type.lower()
        encoded = base64.b64encode(attachment.data).decode("ascii")
        if media in _SUPPORTED_IMAGE_TYPES:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": media,
                        "data": encoded,
                    },
                }
            )
            blocks.append(
                {
                    "type": "text",
                    "text": f"Visual context asset: {attachment.name}",
                }
            )
            added += 1
        elif media == "application/pdf":
            blocks.append(
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/pdf",
                        "data": encoded,
                    }
                }
            )
            added += 1
    return blocks
