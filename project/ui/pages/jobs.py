"""Jobs: every formulation with its state, and every recorded approval."""
import streamlit as st

from project.ui import api, routes
from project.ui.components import job_status
from project.ui.components.spec_view import md

FILTERS = {
    "all": ("All", api.STATES),
    "working": ("Working", ("working",)),
    "needs_input": ("Needs your input", ("needs_input",)),
    "ready": ("Ready", ("ready",)),
    "problems": ("Not supported / Failed", ("unsupported", "failed")),
    "closed": ("Closed", ("approved", "abandoned")),
}
ORIGINS = {"brief": ("From a brief", ":material/edit_note:"), "benchmark": ("Benchmark", ":material/science:")}


def _plain(text: str, limit: int = 90) -> str:
    """Model- or user-written text shown literally in a Markdown label."""
    return md(text if len(text) <= limit else text[:limit - 1] + "…")


def _when(stamp: str) -> str:
    return f"{stamp.replace('T', ' ')[:16]} UTC" if stamp else "unknown"


def _signature(items: list) -> tuple:
    return tuple((f.fid, f.state, f.step) for f in items)


@st.fragment(run_every=3)
def _watch(signature: tuple) -> None:
    st.caption(":material/autorenew: This list refreshes while a formulation is working.")
    if _signature(api.list_formulations()) != signature:
        st.rerun()


st.title("Jobs")
st.write("Every formulation and its state. Open one to answer its questions, fix it, or approve it.")

formulations = api.list_formulations()
counts = {key: sum(f.state in states for f in formulations) for key, (_, states) in FILTERS.items()}
shown = st.query_params.get("show", "all")
initial = shown if shown in FILTERS else next(
    (key for key, (_, states) in FILTERS.items() if key != "all" and shown in states), "all")
choice = st.segmented_control("Show", list(FILTERS), default=initial, required=True, key="jobs_filter",
                              format_func=lambda key: f"{FILTERS[key][0]} ({counts[key]})") or "all"
if choice == "all":
    st.query_params.pop("show", None)
else:
    st.query_params["show"] = choice

if any(f.state == "working" for f in formulations):
    _watch(_signature(formulations))

items = [f for f in formulations if f.state in FILTERS[choice][1]]
if not formulations:
    st.info("No formulations yet. Describe a new problem, or use a benchmark to go straight to approval.")
    with st.container(horizontal=True):
        st.page_link(routes.NEW_PROBLEM, label="Describe a new problem", icon=":material/edit_note:")
        st.page_link(routes.BENCHMARKS, label="Use a benchmark", icon=":material/science:")
elif not items:
    st.caption("No formulations in this group.")
for f in items:
    origin, icon = ORIGINS.get(f.origin, (f.origin, ":material/description:"))
    with st.container(border=True):
        left, middle, right = st.columns([6, 2, 3], vertical_alignment="center")
        with left:
            st.page_link(routes.FORMULATION, label=_plain(f.title), icon=icon, query_params={"fid": f.fid})
            if f.step:
                st.caption(f.step)
        with middle:
            job_status.badge(f.state)
        right.caption(f"{origin} · updated {_when(f.updated_at)} · id {f.fid[:8]}")

st.subheader("Approvals")
approvals = api.list_approvals()
if not approvals:
    st.caption("No approvals recorded yet. Approve a ready formulation from its page, or several together in Batch.")
for approval in approvals:
    titles = [e.title for e in approval.entries] or ["Approval"]
    title = _plain(titles[0], 70) + (f" and {len(titles) - 1} more" if len(titles) > 1 else "")
    kind = "Single plan" if approval.kind == "single" else f"Batch of {len(approval.entries)} plans"
    with st.container(border=True):
        left, right = st.columns([6, 5], vertical_alignment="center")
        left.page_link(routes.APPROVAL, label=title, icon=":material/verified:",
                       query_params={"aid": approval.approval_id})
        right.caption(f"{kind} · approved by {_plain(approval.approver)} · {_when(approval.approved_at)}")
if approvals:
    st.caption("Each approved plan starts at most one solver run.")
