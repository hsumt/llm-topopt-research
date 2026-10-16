"""The one state badge used on every page."""
import streamlit as st

LABELS: dict[str, tuple[str, str]] = {
    "working": ("Working", "blue"),
    "needs_input": ("Needs your input", "orange"),
    "ready": ("Ready for approval", "green"),
    "unsupported": ("Not supported", "violet"),
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
