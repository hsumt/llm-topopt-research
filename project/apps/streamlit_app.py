"""Streamlit front end for solver-independent engineering problem formulation.

Run from repository root:

    streamlit run project/apps/streamlit_app.py
"""

from __future__ import annotations

import base64

import streamlit as st
import streamlit.components.v1 as components

from project.apps.spec_card_examples import SPEC_CARD_EXAMPLES
from project.formulation.context import ContextAttachment, extract_attachment_text
from project.formulation.preview import (
    context_bullets,
    open_issue_bullets,
    problem_bullets,
    problem_graph_dot,
)
from project.formulation.session import approve_session, continue_session, start_session
from project.formulation.verification import check_readiness
from project.paths import ARTIFACT_ROOT


st.set_page_config(
    page_title="Engineering Problem Formulation",
    page_icon="⚙️",
    layout="wide",
)


def _initialize() -> None:
    defaults = {
        "session": None,
        "last_error": None,
        "attachments": [],
        "context_draft": "",
        "problem_draft": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _reset() -> None:
    for key in [
        "session",
        "last_error",
        "attachments",
        "context_draft",
        "problem_draft",
    ]:
        if key in st.session_state:
            del st.session_state[key]
    _initialize()


def _uploaded_to_attachments(uploaded_files) -> list[ContextAttachment]:
    result: list[ContextAttachment] = []
    for item in uploaded_files or []:
        result.append(
            ContextAttachment(
                name=item.name,
                media_type=item.type or "application/octet-stream",
                data=item.getvalue(),
            )
        )
    return result


def _render_attachment(attachment: ContextAttachment) -> None:
    media = attachment.media_type.lower()
    st.markdown(f"**{attachment.name}** · `{attachment.media_type}`")
    if media.startswith("image/"):
        st.image(attachment.data, use_container_width=True)
        st.caption(
            "Reference this filename, visible labels, colors, regions, or directions in your clarification answers."
        )
        return

    if media == "application/pdf":
        encoded = base64.b64encode(attachment.data).decode("ascii")
        components.html(
            f"""
            <object data="data:application/pdf;base64,{encoded}"
                    type="application/pdf" width="100%" height="520px">
              <p>PDF preview unavailable in this browser.</p>
            </object>
            """,
            height=540,
            scrolling=True,
        )
        text = extract_attachment_text(attachment)
        if text.strip():
            with st.expander("Extracted PDF text used for context retrieval"):
                st.text(text[:12000])
        st.download_button(
            f"Download {attachment.name}",
            data=attachment.data,
            file_name=attachment.name,
            mime="application/pdf",
            key=f"download_{attachment.name}",
        )
        return

    text = extract_attachment_text(attachment)
    if text.strip():
        st.text(text[:12000])
    else:
        st.caption("This file is retained as context metadata but has no built-in preview.")


def _render_bullets(items: list[str], empty: str) -> None:
    if not items:
        st.caption(empty)
        return
    for item in items:
        st.markdown(f"- {item}")


_initialize()

st.title("Engineering Problem Formulation")
st.caption(
    "Context + problem → semantic routing → solver-independent parsing → "
    "batched formulation clarification → human approval. No solver is selected or executed."
)

session = st.session_state.session

# ---------------------------------------------------------------------------
# New problem screen
# ---------------------------------------------------------------------------
if session is None:
    with st.sidebar:
        st.header("Pilot examples")
        example_name = st.selectbox(
            "Load a specification-card prompt",
            ["(none)"] + list(SPEC_CARD_EXAMPLES.keys()),
        )
        if st.button("Load selected example", use_container_width=True):
            if example_name != "(none)":
                selected = SPEC_CARD_EXAMPLES[example_name]
                st.session_state.context_draft = selected["context"]
                st.session_state.problem_draft = selected["problem"]
                st.rerun()

    st.subheader("Start a new problem")
    st.write(
        "Provide project/background context separately from the engineering request. "
        "Upload drawings, screenshots, or PDFs when geometry/regions are easier to show than describe."
    )

    with st.form("start_formulation"):
        context = st.text_area(
            "Context (optional)",
            value=st.session_state.context_draft,
            height=180,
            placeholder=(
                "Background about the part, environment, manufacturing, interfaces, "
                "prior decisions, operating conditions, etc."
            ),
        )
        problem = st.text_area(
            "Engineering problem / question",
            value=st.session_state.problem_draft,
            height=180,
            placeholder=(
                "Example: Design a lightweight 3-D bracket while keeping displacement below 0.5 mm."
            ),
        )
        uploaded = st.file_uploader(
            "Supporting context files",
            type=["png", "jpg", "jpeg", "webp", "gif", "pdf", "txt", "md"],
            accept_multiple_files=True,
            help=(
                "Images/PDFs are shown back to the engineer and supplied as visual context to the parser/critic. "
                "PDF/text content is also searched before clarification questions are asked."
            ),
        )
        submitted = st.form_submit_button("Start formulation", type="primary")

    if submitted:
        if not problem.strip():
            st.error("Enter an engineering problem first.")
        else:
            attachments = _uploaded_to_attachments(uploaded)
            st.session_state.attachments = attachments
            st.session_state.context_draft = context
            st.session_state.problem_draft = problem
            with st.status("Starting formulation...", expanded=True) as status:
                def progress(message: str) -> None:
                    status.write(message)

                try:
                    st.session_state.session = start_session(
                        problem=problem,
                        context=context or None,
                        attachments=attachments,
                        progress=progress,
                    )
                    st.session_state.last_error = None
                    status.update(
                        label="Initial formulation review complete",
                        state="complete",
                        expanded=False,
                    )
                    st.rerun()
                except Exception as exc:
                    st.session_state.last_error = str(exc)
                    status.update(
                        label="Formulation start failed",
                        state="error",
                        expanded=True,
                    )
                    st.error(f"Could not start formulation: {exc}")
                    debug_path = ARTIFACT_ROOT / "debug" / "parser_last_failure.json"
                    if debug_path.exists():
                        st.warning(
                            "The exact failed parser response was saved locally. "
                            "Download it before running the model again so the next fix "
                            "can be made without another paid parser call."
                        )
                        st.download_button(
                            "Download parser failure snapshot",
                            data=debug_path.read_bytes(),
                            file_name="parser_last_failure.json",
                            mime="application/json",
                            key="download_parser_failure",
                        )

    st.stop()

# ---------------------------------------------------------------------------
# Active session
# ---------------------------------------------------------------------------
session = st.session_state.session
attachments: list[ContextAttachment] = st.session_state.attachments
readiness = check_readiness(session)

with st.sidebar:
    st.header("Session")
    st.write(f"**Status:** `{session.status}`")
    st.write(f"**Revision:** {session.revision}")
    st.write(f"**Blocking items:** {len(readiness.blockers)}")
    st.write(f"**Context assets:** {len(attachments)}")

    if st.button("New problem", use_container_width=True):
        _reset()
        st.rerun()

    st.download_button(
        "Download session JSON",
        data=session.model_dump_json(indent=2),
        file_name="formulation_session.json",
        mime="application/json",
        use_container_width=True,
    )

    if session.usage:
        with st.expander("LLM usage"):
            total = sum(int(item.get("total_tokens", 0)) for item in session.usage)
            st.metric("Recorded tokens", total)
            st.json(session.usage)

if st.session_state.last_error:
    st.error(st.session_state.last_error)

status_col, approval_col = st.columns([3, 1])
with status_col:
    if session.status == "approved":
        st.success("Formulation approved. Solver handoff is intentionally disabled in this build.")
    elif session.status == "cancelled":
        st.warning("Formulation session cancelled.")
    elif session.status == "ready_for_approval":
        st.success(
            "No blocking pre-solve issues remain. Compare the visual/text mirrors with your intent before approval."
        )
    else:
        st.info(
            f"Formulation in progress — {len(readiness.blockers)} blocking item(s) remain."
        )

with approval_col:
    if session.status == "ready_for_approval":
        if st.button("Approve formulation", type="primary", use_container_width=True):
            try:
                st.session_state.session = approve_session(session)
                st.session_state.last_error = None
                st.rerun()
            except Exception as exc:
                st.session_state.last_error = str(exc)
                st.rerun()

main_col, mirror_col = st.columns([1.25, 1.0], gap="large")

# ---------------------------------------------------------------------------
# Conversation + batch clarifications
# ---------------------------------------------------------------------------
with main_col:
    st.subheader("Formulation conversation")

    for message in session.messages:
        if message.role == "system":
            continue
        with st.chat_message(message.role):
            st.markdown(message.content)

    packet = session.critic_result.clarification_packet.questions
    if session.status not in {"approved", "cancelled", "ready_for_approval"} and packet:
        st.markdown("---")
        st.subheader(f"Clarification packet · {len(packet)} item(s)")
        st.caption(
            "Answer the independent items you can resolve now, then submit once. "
            "Leave an item blank if you genuinely do not know it yet."
        )

        with st.form(f"clarification_packet_{session.revision}"):
            answers: dict[str, str] = {}
            for index, question in enumerate(packet, start=1):
                st.markdown(f"#### {index}. {question.prompt}")
                st.caption(question.why_needed)
                if question.related_context_ids:
                    st.caption(
                        "Relevant context: " + ", ".join(question.related_context_ids)
                    )
                if question.depends_on:
                    st.caption("Depends on: " + ", ".join(question.depends_on))

                key = f"packet_{session.revision}_{question.id}"
                if question.answer_type == "single_choice" and question.options:
                    options = ["— choose —"] + question.options
                    selected = st.selectbox(
                        "Answer",
                        options,
                        key=key,
                        label_visibility="collapsed",
                    )
                    answers[question.id] = "" if selected == "— choose —" else selected
                elif question.answer_type == "multiple_choice" and question.options:
                    selected = st.multiselect(
                        "Answer",
                        question.options,
                        key=key,
                        label_visibility="collapsed",
                    )
                    answers[question.id] = ", ".join(selected)
                else:
                    placeholder = (
                        "Reference an uploaded filename/color/label/region if helpful..."
                        if question.answer_type == "visual_reference"
                        else "Your engineering decision..."
                    )
                    answers[question.id] = st.text_area(
                        "Answer",
                        key=key,
                        height=90,
                        placeholder=placeholder,
                        label_visibility="collapsed",
                    )
                st.divider()

            submit_packet = st.form_submit_button(
                "Submit clarification packet",
                type="primary",
                use_container_width=True,
            )

        if submit_packet:
            if not any(value.strip() for value in answers.values()):
                st.warning("Answer at least one clarification item before submitting.")
            else:
                with st.status("Applying clarification packet...", expanded=True) as status:
                    def progress(message: str) -> None:
                        status.write(message)

                    try:
                        st.session_state.session = continue_session(
                            session,
                            structured_answers=answers,
                            attachments=attachments,
                            progress=progress,
                        )
                        st.session_state.last_error = None
                        status.update(
                            label="Clarification round complete",
                            state="complete",
                            expanded=False,
                        )
                    except Exception as exc:
                        st.session_state.last_error = str(exc)
                        status.update(
                            label="Clarification update failed",
                            state="error",
                            expanded=True,
                        )
                st.rerun()

    if session.status not in {"approved", "cancelled"}:
        reply = st.chat_input(
            "Add a correction, answer in free form, or reference an uploaded drawing..."
        )
        if reply:
            with st.status("Processing clarification...", expanded=True) as status:
                def progress(message: str) -> None:
                    status.write(message)

                try:
                    st.session_state.session = continue_session(
                        session,
                        reply,
                        attachments=attachments,
                        progress=progress,
                    )
                    st.session_state.last_error = None
                    status.update(
                        label="Clarification round complete",
                        state="complete",
                        expanded=False,
                    )
                except Exception as exc:
                    st.session_state.last_error = str(exc)
                    status.update(
                        label="Clarification update failed",
                        state="error",
                        expanded=True,
                    )
            st.rerun()

# ---------------------------------------------------------------------------
# Always-visible human mirrors of current problem/context
# ---------------------------------------------------------------------------
with mirror_col:
    st.subheader("What the AI is currently formulating")

    overview_tab, context_tab, visuals_tab, spec_tab, issues_tab, prov_tab, history_tab = st.tabs(
        ["Problem", "Context", "Visuals", "Spec", "Issues", "Provenance", "History"]
    )

    with overview_tab:
        st.markdown("### Original request")
        st.info(session.original_problem)

        st.markdown("### Current interpreted problem")
        _render_bullets(
            problem_bullets(session),
            "No structured formulation has been extracted yet.",
        )

        st.markdown("### Problem map")
        try:
            st.graphviz_chart(problem_graph_dot(session), use_container_width=True)
        except Exception:
            st.caption("Problem map could not be rendered in this Streamlit environment.")

        st.markdown("### Open issues")
        _render_bullets(open_issue_bullets(session), "No current blocking issues.")

    with context_tab:
        st.markdown("### Supplied context")
        if session.supplied_context:
            st.info(session.supplied_context)
        else:
            st.caption("No typed context supplied.")

        st.markdown("### Context interpretation")
        _render_bullets(context_bullets(session), "No context candidates extracted.")

        st.markdown("### Retrieved before the latest review")
        if session.retrieved_context:
            for item in session.retrieved_context:
                with st.expander(f"{item.source} · score {item.score:.2f}"):
                    st.write(item.text)
        else:
            st.caption("No searchable context snippets were retrieved.")

    with visuals_tab:
        if not attachments:
            st.caption("No visual/context files were uploaded for this session.")
        else:
            st.caption(
                "These are the same user-supplied assets available to the formulation model. "
                "Use filenames/labels/colors in clarification answers when needed."
            )
            for attachment in attachments:
                with st.expander(attachment.name, expanded=attachment.media_type.startswith("image/")):
                    _render_attachment(attachment)

    with spec_tab:
        st.json(session.parser_result.spec.model_dump(exclude_none=True))

    with issues_tab:
        if readiness.blockers:
            st.markdown("**Blocking**")
            for blocker in readiness.blockers:
                st.error(blocker)
        if readiness.warnings:
            st.markdown("**Warnings**")
            for warning in readiness.warnings:
                st.warning(warning)
        if not readiness.blockers and not readiness.warnings:
            st.success("No current pre-solve issues.")

        if packet:
            st.markdown("### Current clarification packet")
            for q in packet:
                st.markdown(f"- **{q.id}:** {q.prompt}")

    with prov_tab:
        if not session.parser_result.field_provenance:
            st.caption("No provenance records.")
        for item in session.parser_result.field_provenance:
            with st.expander(item.field_path):
                st.write(f"**Source:** {item.source}")
                st.write(f"**Value:** `{item.value}`")
                st.write(f"**Evidence:** {item.evidence}")
                st.write(f"**Confidence:** {item.confidence:.2f}")

    with history_tab:
        if not session.revisions:
            st.caption("No formulation revisions yet.")
        for revision in session.revisions:
            with st.expander(f"Revision {revision.revision}"):
                st.write(f"**User clarification:** {revision.user_message}")
                if revision.structured_answers:
                    st.write("**Batch answers:**")
                    st.json(revision.structured_answers)
                if revision.operations:
                    st.write("**Applied patch:**")
                    st.json([operation.model_dump() for operation in revision.operations])
                if revision.incorporated_context_ids:
                    st.write(
                        "**Context incorporated:** "
                        + ", ".join(revision.incorporated_context_ids)
                    )
