"""Blocking issues from the pre-solve check, grouped by what resolves them."""
import streamlit as st

from project.ui.components.spec_view import md

GROUPS = (
    ("decision", "Decisions needed"),
    ("data", "Missing data"),
)
ORIGINS = {
    "code": ":gray-badge[:material/code: checked by code]",
    "model": ":gray-badge[:material/smart_toy: raised by the reviewer model]",
}


def render(blockers: dict[str, list]) -> None:
    for kind, heading in GROUPS:
        items = blockers.get(kind) or []
        if not items:
            continue
        st.markdown(f"**{heading}** ({len(items)})")
        st.markdown("\n".join(f"- {md(item.description)} {ORIGINS.get(item.origin, '')}" for item in items))
