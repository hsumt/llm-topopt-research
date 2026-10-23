"""Blocking issues from the pre-solve check, grouped by what resolves them."""
import streamlit as st

from project.ui.components.spec_view import md

# Role colour and Material icon per kind of issue: one meaning per colour, always with its icon (colour directives
# follow the active theme). Both groups need a decision from the engineer, so both are "needs decision".
NEEDS_DECISION = ("orange", "help")
UNSUPPORTED = ("red", "block")
GROUPS = (
    ("decision", "Decisions needed", NEEDS_DECISION),
    ("data", "Missing data", NEEDS_DECISION),
)
ORIGINS = {
    "code": ":gray-badge[:material/code: checked by code]",
    "model": ":gray-badge[:material/smart_toy: raised by the reviewer model]",
}


def mark(role: tuple[str, str], text: str = "") -> str:
    """The role's icon, then `text` (trusted markdown, never model text), in the role colour."""
    colour, icon = role
    return f":{colour}[:material/{icon}:{' ' + text if text else ''}]"


def stated(item) -> str:
    """What the spec states at the item's field, quoted (model or engineer text, escaped); "" when nothing."""
    return f" :small[(stated: “{md(item.stated)}”)]" if getattr(item, "stated", None) else ""


def heading(role: tuple[str, str], title: str, count: int) -> None:
    st.markdown(mark(role, f"**{title}** ({count})"))


def render(blockers: dict[str, list]) -> None:
    for kind, title, role in GROUPS:
        items = blockers.get(kind) or []
        if not items:
            continue
        heading(role, title, len(items))
        st.markdown("\n".join(f"- {mark(role)} {md(item.description)}{stated(item)} {ORIGINS.get(item.origin, '')}"
                              for item in items))
