"""Approve exact plans for ready formulations, or read a recorded approval."""
import hashlib

import streamlit as st

from project.ui import api, routes
from project.ui.components import job_status, plan_view
from project.ui.components.spec_view import md


def _links() -> None:
    with st.container(horizontal=True):
        st.page_link(routes.JOBS, label="Go to Jobs", icon=":material/list_alt:")
        st.page_link(routes.BATCH, label="Go to Batch", icon=":material/stacks:")


def _open(fid: str) -> None:
    st.page_link(routes.FORMULATION, label="Open this formulation", icon=":material/open_in_new:",
                 query_params={"fid": fid})


def _run_line(approval, entry) -> None:
    """This plan's run: a Run button while the approval has not started it, otherwise its state."""
    fid = entry.fid
    notice = st.session_state.get("run_notices", {}).pop(fid, None)
    if notice:
        st.error(f"Approved, but the run did not start: {notice}")
    formulation = api.get(fid)
    if formulation is None or formulation.approval_id != approval.approval_id:
        st.caption("A later approval replaced this one for the next run of this formulation.")
        return
    actions = api.run_actions(fid)
    if actions["can_run"]:
        if st.button("Run this plan", key=f"run_{fid}", type="primary", icon=":material/play_arrow:",
                     help="Starts the solver on exactly this plan. Each approval starts at most one run."):
            try:
                api.launch(fid, approval.approval_id)
            except (ValueError, RuntimeError, OSError) as error:
                st.error(str(error))
            else:
                st.switch_page(routes.FORMULATION, query_params={"fid": fid})
    elif actions["run_state"] and not actions["pending"]:
        job_status.run_badge(actions["run_state"])
        st.caption(f"Run {actions['run_state']}. Open the formulation for its results.")


def _recorded(aid: str) -> None:
    approval = api.get_approval(aid)
    if approval is None:
        st.title("Approval")
        st.write("This page shows a recorded approval. Open one from Jobs.")
        st.warning("No approval with this id was found.")
        _links()
        st.stop()
    count = len(approval.entries)
    st.title("Approved plan" if count == 1 else f"Approved batch of {count} plans")
    st.write("This is the record of what was approved; it does not change.")
    st.caption(f"Approved by {md(approval.approver)} on {job_status.when(approval.approved_at)} · "
               f"{'single plan' if approval.kind == 'single' else 'batch'} · id {approval.approval_id}")
    st.caption("Each approved plan starts at most one solver run.")
    for entry in approval.entries:
        with st.container(border=True):
            st.subheader(md(entry.title))
            _run_line(approval, entry)
            _open(entry.fid)
            plan_view.render(entry)


def _review(fids: list[str]) -> None:
    st.title("Review and approve")
    st.write("Check every input below. Approving records these exact plans.")
    items, hashes = [], []
    for fid in fids:
        formulation = api.get(fid)
        with st.container(border=True):
            if formulation is None:
                st.warning(f"No formulation with id {fid} was found. It is left out.")
                continue
            st.subheader(md(formulation.title))
            with st.container(horizontal=True, vertical_alignment="center"):
                job_status.badge(formulation.state)
                st.caption(f"id {fid}")
            if formulation.state == "approved":
                actions = api.run_actions(fid)
                if not actions["can_reapprove"]:
                    st.info("Its approved plan is running. It is left out; approve it again after the run ends.")
                    _open(fid)
                    continue
                st.caption("Already approved. Approving again replaces that approval: the next run uses the plan "
                           "approved here, and earlier runs are kept.")
                if formulation.approval_id:
                    st.page_link(routes.APPROVAL, label="Open the current approval record",
                                 icon=":material/receipt_long:", query_params={"aid": formulation.approval_id})
            elif formulation.state != "ready":
                st.warning(f"Not ready for approval ({job_status.label(formulation.state).lower()}). It is left out.")
                _open(fid)
                continue
            budget = st.columns([1, 1, 2])
            iterations = int(budget[0].number_input(
                "Max iterations", min_value=1, max_value=api.MAX_ITERATIONS, value=api.MAX_ITERATIONS, step=1,
                key=f"iterations_{fid}",
                help=f"The most optimization iterations the solver may run. {api.MAX_ITERATIONS} is a full run; "
                     "use a few iterations for a quick check."))
            ranks = int(budget[1].number_input(
                "MPI processes", min_value=1, max_value=api.MAX_RANKS, value=1, step=1, key=f"ranks_{fid}",
                help="Parallel solver processes."))
            try:
                entry = api.plan_entry(fid, iterations, ranks)
            except (ValueError, RuntimeError) as error:
                st.warning(f"{error} It is left out.")
                _open(fid)
                continue
            plan_view.render(entry)
            items.append((fid, iterations, ranks, entry.plan_hash))
            hashes.append(entry.plan_hash)

    st.divider()
    if not items:
        st.info("Nothing on this page can be approved. Ready and approved formulations are listed in Jobs.")
        _links()
        return
    count = len(items)
    st.subheader("Approve")
    approver = st.text_input("Your name", value=st.session_state.get("approver", ""), key="approver_name",
                             placeholder="Recorded with the approval")
    st.session_state["approver"] = approver.strip()
    # The key follows the plans shown, so any change to a plan clears the confirmation.
    shown = hashlib.sha256(",".join(hashes).encode()).hexdigest()[:16]
    confirmed = st.checkbox("I approve this plan exactly as shown" if count == 1
                            else f"I approve these {count} plans exactly as shown", key=f"confirm_{shown}",
                            help="Changing a budget above clears this tick, so you always approve what is shown.")
    allowed = bool(approver.strip()) and confirmed
    hint = None if allowed else "Enter your name and tick the box above first."
    with st.container(horizontal=True):
        run_now = st.button("Approve and run" if count == 1 else f"Approve and run all {count}", type="primary",
                            icon=":material/play_arrow:", disabled=not allowed, help=hint)
        record_only = st.button("Approve only", icon=":material/verified:", disabled=not allowed, help=hint)
    st.caption("Approve and run records the plans, then starts each solver run on exactly what was approved. "
               "Approve only records them; run each later from its approval record or formulation page.")
    if allowed and (run_now or record_only):
        try:
            if run_now:
                aid, outcomes = api.approve_and_run(items, approver)
            else:
                aid, outcomes = api.approve(items, approver), {}
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
            return
        failed = {fid: outcome["error"] for fid, outcome in outcomes.items() if "error" in outcome}
        st.session_state.setdefault("run_notices", {}).update(failed)
        if run_now and count == 1:
            # One plan: go straight to its run.
            st.switch_page(routes.FORMULATION, query_params={"fid": items[0][0]})
        st.switch_page(routes.APPROVAL, query_params={"aid": aid})


aid = st.query_params.get("aid")
fids = list(dict.fromkeys(part.strip() for part in (st.query_params.get("fids") or "").split(",") if part.strip()))
if aid:
    _recorded(aid)
elif fids:
    _review(fids)
else:
    st.title("Approval")
    st.write("Approve formulations here: open a ready or approved one from Jobs, or select several in Batch.")
    _links()
