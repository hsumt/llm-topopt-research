"""Requirements the connected solver cannot express: future work for the codebase."""
import streamlit as st

from project.ui.components.issues_view import ORIGINS, UNSUPPORTED, heading, mark, stated
from project.ui.components.spec_view import md


def _field(path: str | None) -> str:
    return f"field `{path.replace('`', '')}`" if path else "field not identified"


def render(blockers: list) -> None:
    if not blockers:
        return
    heading(UNSUPPORTED, "Not supported by the connected solver", len(blockers))
    st.markdown("\n".join(f"- {mark(UNSUPPORTED)} {md(b.description)} ({_field(b.field)}){stated(b)} {ORIGINS.get(b.origin, '')}"
                          for b in blockers))
    st.caption("The connected solver cannot run these requirements. "
               "In a later version each one becomes a gap record for the codebase.")
