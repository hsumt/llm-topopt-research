"""The state badges, origin labels and timestamps used on every page."""
from datetime import datetime

import streamlit as st

LABELS: dict[str, tuple[str, str]] = {
    "working": ("Working", "blue"),
    "needs_input": ("Needs your input", "orange"),
    "ready": ("Ready for approval", "green"),
    "unsupported": ("Not supported", "red"),   # the role colour for unsupported; Failed differs by label and icon
    "failed": ("Failed", "red"),
    "approved": ("Approved", "gray"),
    "abandoned": ("Abandoned", "gray"),
}

_ICONS = {
    "working": ":material/progress_activity:",
    "needs_input": ":material/help:",
    "ready": ":material/check_circle:",
    "unsupported": ":material/block:",
    "failed": ":material/error:",
    "approved": ":material/verified:",
    "abandoned": ":material/archive:",
}


def label(state: str) -> str:
    return LABELS.get(state, (state.replace("_", " ").capitalize(), "gray"))[0]


def badge(state: str) -> None:
    st.badge(label(state), icon=_ICONS.get(state), color=LABELS.get(state, ("", "gray"))[1])


def run_badge(run_state: str) -> None:
    """A solver run's state, shown with the formulation badge of the same meaning."""
    badge({"completed": "approved", "failed": "failed", "interrupted": "failed"}.get(run_state, "working"))


ORIGINS = {"brief": ("From a brief", ":material/edit_note:"),
           "benchmark": ("Benchmark", ":material/science:"),
           "edit": ("Copy of an approved formulation", ":material/edit_document:")}


def origin(kind: str) -> tuple[str, str]:
    """(label, icon) for a formulation's origin."""
    return ORIGINS.get(kind, (kind, ":material/description:"))


def when(stamp: str) -> str:
    try:
        return datetime.fromisoformat(stamp).strftime("%Y-%m-%d %H:%M UTC")
    except ValueError:
        return stamp or "unknown time"
