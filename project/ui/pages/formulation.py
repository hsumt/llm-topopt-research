"""One formulation: its specification, the pre-solve check, and what it needs next."""
import matplotlib.pyplot as plt
import streamlit as st

from project.ui import api, routes
from project.ui.components import (failure_view, flow_canvas, gap_report, issues_view, job_status, plan_view,
                                   result_view, questions_form, spec_view, usage_view)
from project.ui.components.spec_view import md

NEXT = {
    "working": "The system is working on this formulation in the background; this page updates by itself.",
    "needs_input": "The specification is incomplete: answer the questions below or add information.",
    "ready": "The pre-solve check passed: check the specification and solver blocks below, then review and approve.",
    "unsupported": ("The connected solver cannot run this specification as stated: change the requirement under "
                    "\"Add information or a correction\" below, or abandon it."),
    "failed": "The last step did not finish: read what happened below, then retry.",
    "approved": "This formulation is approved; its approval record holds the exact plan.",
    "abandoned": "This formulation was abandoned and is kept for reference only.",
}
NO_QUESTIONS = "The specification is incomplete: add the missing information below."
EDITABLE = ("needs_input", "unsupported", "ready")


def _label(key: str, prompts: dict[str, str]) -> str:
    return "Added information or correction" if key == api.ADDED_INFORMATION_KEY else prompts.get(key, key)


@st.fragment(run_every=2)
def _watch(fid: str, updated_at: str) -> None:
    current = api.get(fid)
    if current is None or current.state != "working" or current.updated_at != updated_at:
        st.rerun()


def _geometry(fid: str, session) -> None:
    spec = session.spec
    st.subheader("Geometry preview")
    preview = api.element_preview(session)
    if preview is not None:
        result_view.render(preview["density"], cut_cells=preview["cut_cells"], key=f"preview_{fid}",
                           title=f"Starting design · {preview['mesh'][0]} × {preview['mesh'][1]} elements")
        st.caption("One cube per element: the design the solver starts from, computed by ParaLeSTO from the "
                   "solver inputs (identical to the run's iteration 0).")
    else:
        try:
            figure = api.preview(spec)
        except Exception:
            figure = None
        if figure is None:
            st.caption("Not enough geometry is stated to draw a preview.")
        else:
            st.pyplot(figure, width="stretch")
            plt.close(figure)
    with st.expander("Problem structure"):
        st.graphviz_chart(api.graph(spec), width="stretch")
    st.page_link(routes.FLOW, label="See how this was derived (process flow)", icon=":material/account_tree:",
                 query_params={"fid": fid})


def _precheck(formulation, check) -> None:
    fid, state, session = formulation.fid, formulation.state, formulation.session
    st.subheader("Pre-solve check")
    if check.ready:
        st.success("Every required input is stated and the connected solver can run this specification.")
        plan_view.block_table(check.block_config)
        if state == "ready" and st.button("Review and approve", type="primary", icon=":material/task_alt:"):
            st.switch_page(routes.APPROVAL, query_params={"fids": fid})
        return
    issues_view.render(check.blockers)
    gap_report.render(check.blockers.get("unsupported") or [])
    issues = any(check.blockers.values())
    if not issues and not check.questions:
        st.info("The check did not pass, but no specific issue was recorded."
                + (" Add information or a correction below." if state in EDITABLE else ""))
    if state != "needs_input":
        return
    if not check.questions:
        if issues:
            st.caption("No questions were generated for these issues. "
                       "Use \"Add information or a correction\" below to resolve them.")
        return
    st.subheader("Questions")
    _walkthrough(fid, check.questions, len(session.revisions))
    answers = questions_form.render(fid, check.questions, len(session.revisions))
    if answers is not None:
        try:
            api.answer(fid, answers)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
        else:
            st.rerun()


def _walkthrough(fid: str, questions, round_index: int) -> None:
    """Prepared answers for walkthrough examples: fill the matching answer boxes in one click."""
    prepared = api.walkthrough_answers(fid)
    if not prepared:
        return
    matching = {q.key: prepared[q.key] for q in questions if q.key in prepared}

    def fill() -> None:
        for key, value in matching.items():
            st.session_state[questions_form.answer_key(fid, round_index, key)] = value

    with st.expander("Walkthrough: prepared answers", expanded=True, icon=":material/school:"):
        st.caption("This brief is a walkthrough example with answers prepared in advance. "
                   "Answers for code-check questions can be filled automatically; reviewer questions vary, "
                   "so their prepared answers are shown for you to copy if they fit.")
        for key, value in prepared.items():
            st.markdown(f"**{md(key)}**: {md(value)}")
        st.button("Fill prepared answers", on_click=fill, disabled=not matching, icon=":material/edit:",
                  help="Puts the prepared answers into the matching boxes below. Review them before submitting.")


def _run(fid: str, record) -> None:
    """Run the approved plan and show its live status and results."""
    st.subheader("Run")
    notice = st.session_state.get("run_notices", {}).pop(fid, None)
    if notice:
        st.error(f"Approved, but the run did not start: {notice}")
    actions = api.run_actions(fid)
    info = api.run_info(fid)
    if actions["can_run"]:
        entry = next((e for e in record.entries if e.fid == fid), None) if record else None
        budget = entry.budget if entry else {}
        st.write(f"The approved plan has not run yet: {plan_view.budget_text(budget)}.")
        with st.container(horizontal=True, vertical_alignment="center"):
            if st.button("Run approved plan", type="primary", icon=":material/play_arrow:"):
                try:
                    api.launch(fid)
                except (ValueError, RuntimeError, OSError) as error:
                    st.error(str(error))
                else:
                    st.rerun()
            _approve_again(fid, "Change the budget (approve again)")
    if info is None:
        return
    st.page_link(routes.FLOW, label="Open the full process flow", icon=":material/account_tree:",
                 query_params={"fid": fid})
    if info["pending"]:
        st.caption("Previous run, from an earlier approval:")
    if info["state"] in ("starting", "running"):
        _run_live(fid)
    else:
        _run_result(fid, info)
        if not info["pending"] and actions["can_reapprove"]:
            _approve_again(fid, "Approve another run")


def _process(fid: str) -> None:
    """The formulation's process flow, opened when a step or run starts and following the active step."""
    live, key = api.is_live(fid), f"flow_open_{fid}"
    # Open it once when something starts; after that, open or closed is the engineer's choice.
    if live and not st.session_state.get(key + "_opened"):
        st.session_state[key] = st.session_state[key + "_opened"] = True
    elif not live:
        st.session_state.pop(key + "_opened", None)
    with st.expander("Process flow · live" if live else "Process flow", key=key, on_change="rerun",
                     icon=":material/account_tree:"):
        if live or st.session_state.get(key):   # built only when it can be seen
            flow_canvas.show(fid, f"page_{fid}", height=520)


def _approve_again(fid: str, label: str) -> None:
    st.page_link(routes.APPROVAL, label=label, icon=":material/task_alt:", query_params={"fids": fid})


@st.fragment(run_every=3)
def _run_live(fid: str) -> None:
    info = api.run_info(fid)
    if info is None or info["state"] not in ("starting", "running"):
        st.rerun()   # the whole page: a finished run is drawn once, without refreshing
    _run_result(fid, info)
    st.caption(("ParaView is rendering the result. " if info["paraview"]["rendering"] else "")
               + "Refreshing every 3 s while the run is active.")


def _run_result(fid: str, info: dict) -> None:
    rows = info["history"]
    budget = info.get("budget")
    with st.container(horizontal=True, vertical_alignment="center"):
        job_status.run_badge(info["state"])
        st.write(f"**{info['state'].capitalize()}** · {len(rows)}{f' of {budget}' if budget else ''} iterations "
                 f"recorded · run {info['run']}")
    if info.get("reason"):
        st.warning(info["reason"])
    if info.get("log_tail"):
        with st.expander("Solver log (last lines)", expanded=not info.get("reason")):
            st.code(info["log_tail"], language=None)
    if rows:
        last = rows[-1]
        cols = st.columns(3)
        cols[0].metric("Volume fraction", f"{last['volume_fraction']:.3f}",
                       help=f"Limit {info['config'].get('volume_fraction')}")
        cols[1].metric("Compliance", f"{last['compliance']:.4g}")
        limit = info["config"].get("stress_limit_pa")
        cols[2].metric("Stress p-norm", f"{last['stress_pnorm_pa'] / 1e6:.1f} MPa",
                       help=f"Limit {limit / 1e6:.0f} MPa" if limit else None)
    frames = info.get("frames") or []
    if frames:
        options = [f["iteration"] for f in frames]
        picked = st.select_slider("Design at iteration", options, value=options[-1], key=f"frame_{fid}",
                                  help="Element densities from ParaLeSTO, saved every 10 iterations and at the end.")
        frame = frames[options.index(picked)]
        result_view.render(frame["density"], cut_cells=info["cut_cells"], key=f"{fid}_{picked}",
                           title=f"Iteration {picked} · {frame['boundary_points']} boundary points · ParaLeSTO elements")
        pv_frames = info["paraview"]["frames"]
        if pv_frames:
            pv_options = [f["iteration"] for f in pv_frames]
            nearest = min(pv_options, key=lambda i: abs(i - picked))
            st.image(pv_frames[pv_options.index(nearest)]["path"],
                     caption=f"ParaView render, iteration {nearest}: element cubes from the run's cell-data density "
                             "(no smoothing).")
        elif (info["paraview"]["status"] or {}).get("state") in ("unavailable", "failed"):
            st.caption(f"No ParaView render: {info['paraview']['status'].get('reason')}")
        level_set = (info.get("summary") or {}).get("level_set")
        if level_set:
            st.caption(f"Level set: {level_set['level_set_module']} · optimizer: {level_set['optimizer']} · "
                       f"{level_set['level_set_updates']} level-set updates applied.")
    if rows:
        st.line_chart({"volume fraction": [r["volume_fraction"] for r in rows]}, height=160)
        st.line_chart({"compliance": [r["compliance"] for r in rows]}, height=160)
        st.line_chart({"stress p-norm (MPa)": [r["stress_pnorm_pa"] / 1e6 for r in rows]}, height=160)
    if info["state"] == "completed":
        st.caption("Reaching the iteration budget does not establish convergence.")


def _add_information(fid: str, expanded: bool) -> None:
    with st.expander("Add information or a correction", expanded=expanded):
        st.caption("State a fact or correct the specification in plain language. "
                   "The system applies it and checks the specification again.")
        text = st.text_area("Information or correction", key=f"information_{fid}", label_visibility="collapsed",
                            placeholder="For example: the load acts in the negative y direction.")
        if st.button("Apply", key=f"apply_information_{fid}", icon=":material/edit:"):
            try:
                api.add_information(fid, text)
            except (ValueError, RuntimeError) as error:
                st.error(str(error))
            else:
                st.rerun()


def _history(formulation) -> None:
    session = formulation.session
    with st.expander("History"):
        st.markdown("**Brief**")
        st.text(formulation.brief)
        st.markdown("**Context**")
        if formulation.context:
            st.text(formulation.context)
        else:
            st.caption("None given.")
        if session is None:
            return
        for number, revision in enumerate(session.revisions, 1):
            st.divider()
            st.markdown(f"**Round {number}**")
            prompts = {question.key: question.prompt for question in revision.questions}
            for question in revision.questions:
                st.markdown(md(question.prompt))
                answer = revision.answers.get(question.key)
                if answer:
                    st.text(answer)
                else:
                    st.caption("Not answered.")
            for key, answer in revision.answers.items():
                if key not in prompts:
                    st.markdown(md(_label(key, prompts)))
                    st.text(answer)
            changed = [".".join(str(part) for part in update.path) for update in revision.resolution.updates]
            st.caption(f"Changed in the specification: {', '.join(changed)}" if changed
                       else "No specification fields changed.")
        if session.pending_answers:
            st.divider()
            st.markdown("**Waiting to be applied**")
            st.caption("The step that applies these answers did not finish; Retry applies them again.")
            prompts = {question.key: question.prompt for question in session.review.questions}
            for key, answer in session.pending_answers.items():
                st.markdown(md(_label(key, prompts)))
                st.text(answer)


def _usage(session) -> None:
    with st.expander("Model usage"):
        usage_view.render(session.usage)


def _abandon(fid: str) -> None:
    st.divider()
    with st.popover("Abandon", icon=":material/block:"):
        st.warning("Abandon this formulation? It stays in Jobs for reference but can no longer be answered, "
                   "retried or approved.")
        if st.button("Yes, abandon it", key=f"abandon_yes_{fid}"):
            try:
                api.abandon(fid)
            except ValueError as error:
                st.error(str(error))
            else:
                st.rerun()


fid = st.query_params.get("fid")
formulation = api.get(fid) if fid else None
if formulation is None:
    st.title("Formulation")
    st.write("This page shows one formulation. Open one from Jobs.")
    if fid:
        st.warning("No formulation with this id was found.")
    st.page_link(routes.JOBS, label="Go to Jobs", icon=":material/list_alt:")
    st.stop()

state, session = formulation.state, formulation.session
check = api.precheck(session) if session is not None and state not in ("working", "failed", "approved") else None
st.title(md(formulation.title))
st.write(NO_QUESTIONS if state == "needs_input" and check and not check.questions else NEXT[state])
with st.container(horizontal=True, vertical_alignment="center"):
    job_status.badge(state)
    st.caption(f"{job_status.origin(formulation.origin)[0]} · created {job_status.when(formulation.created_at)}"
               f" · updated {job_status.when(formulation.updated_at)} · id {fid}")

if state == "working":
    st.info(f"{formulation.step or 'Working'}…", icon=":material/progress_activity:")
    _watch(fid, formulation.updated_at)
    _process(fid)
elif state == "failed":
    failure_view.render(session.failure if session else None, formulation.error)
    if st.button("Retry", type="primary", icon=":material/refresh:"):
        try:
            api.retry(fid)
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
        else:
            st.rerun()
    _process(fid)
elif state == "approved":
    record = api.get_approval(formulation.approval_id) if formulation.approval_id else None
    st.success(f"Approved by {md(record.approver)} on {job_status.when(record.approved_at)}." if record else "Approved.",
               icon=":material/verified:")
    if record:
        st.page_link(routes.APPROVAL, label="Open the approval record", icon=":material/receipt_long:",
                     query_params={"aid": record.approval_id})
    _process(fid)
    _run(fid, record)
else:
    _process(fid)

if session is not None and session.parse_completed:
    if state == "working":
        st.caption("Showing the last saved specification. "
                   "The pre-solve check returns when the current step finishes.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        st.subheader("Specification")
        spec_view.render(session.spec)
    with right:
        _geometry(fid, session)
    if state == "failed":
        st.caption("The pre-solve check runs again after a successful retry.")
    elif check is not None:
        _precheck(formulation, check)
        if state in EDITABLE:
            # Open by default when it is the only way forward.
            _add_information(fid, state == "unsupported" or (state == "needs_input" and not check.questions))
elif state != "working":
    st.caption("There is no specification yet: the brief has not been read successfully.")


_history(formulation)
if session is not None:
    _usage(session)
if state not in ("working", "approved", "abandoned"):
    _abandon(fid)
