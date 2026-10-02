from project.examples import EXAMPLES
import streamlit as st
import json
import os
import uuid
from pathlib import Path

from project.execution import ARTIFACTS, launch_run, run_plan, run_status
from project.lbracket import assess_spec, reference_spec
from project.models import Review, Session

from project.drawing import (
    geometry_preview,
    problem_graph,
)
from project.workflow import (
    answer_questions,
    deterministic_blockers,
    ready_to_try,
    retry_review,
    start_session,
)


st.set_page_config(
    page_title="Engineering Formulation",
    page_icon="⚙️",
    layout="wide",
)


if "session" not in st.session_state:
    st.session_state.session = None
st.session_state.setdefault("session_id", None)
st.session_state.setdefault("run_dir", None)


def save_session(session):
    st.session_state.session = session
    if st.session_state.session_id is None:
        st.session_state.session_id = uuid.uuid4().hex
    path = ARTIFACTS / "sessions" / f"{st.session_state.session_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(session.model_dump_json(indent=2))
    temporary.replace(path)


def model_error(error):
    if not os.getenv("ANTHROPIC_API_KEY"):
        st.error("Natural-language clarification needs ANTHROPIC_API_KEY in the container environment. The complete benchmark runs without a model key.")
    else:
        st.error(f"Formulation update failed ({type(error).__name__}). Your previous specification is retained.")

if "request_draft" not in st.session_state:
    st.session_state.request_draft = ""

if "context_draft" not in st.session_state:
    st.session_state.context_draft = ""


st.title("Engineering formulation → 3D L-bracket")
st.caption(
    "Turn an incomplete engineering request into a reviewable problem "
    "before attempting a solver."
)

with st.sidebar:
    st.subheader("Demo problems")
    if st.button("Load complete 3D benchmark"):
        st.session_state.session_id = None
        st.session_state.run_dir = None
        save_session(Session(original_request="Use all documented inputs of the simplified_3D_holes benchmark.",
                             context="Explicit selection of the five-hole reference; calibrated p6 limit, not yield.",
                             spec=reference_spec(), review=Review()))
        st.rerun()
    st.caption("The complete benchmark uses documented physical inputs and needs no model call.")

    example_name = st.selectbox(
        "Load example",
        ["None"] + list(EXAMPLES.keys()),
    )

    if st.button("Load selected example"):
        if example_name != "None":
            example = EXAMPLES[example_name]

            st.session_state.request_draft = example["request"]
            st.session_state.context_draft = example["context"]
            st.session_state.session = None
            st.session_state.session_id = None
            st.session_state.run_dir = None

            st.rerun()

    saved = sorted((ARTIFACTS / "sessions").glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    selected = st.selectbox("Resume saved formulation", ["None", *[p.name for p in saved]])
    if st.button("Resume") and selected != "None":
        try:
            restored = Session.model_validate_json((ARTIFACTS / "sessions" / selected).read_text())
            st.session_state.session_id = Path(selected).stem
            st.session_state.run_dir = None
            save_session(restored)
            st.rerun()
        except (ValueError, OSError):
            st.error("The saved file is not a valid formulation session.")
    uploaded = st.file_uploader("Import formulation JSON", type=["json"])
    if uploaded is not None and st.button("Import formulation"):
        try:
            restored = Session.model_validate_json(uploaded.getvalue())
            st.session_state.session_id = None
            st.session_state.run_dir = None
            save_session(restored)
            st.rerun()
        except ValueError:
            st.error("The file is not a valid formulation Session JSON.")

session = st.session_state.session


if session is None:
    request = st.text_area(
        "Describe the engineering problem",
        key="request_draft",
        height=180,
        placeholder="Example: Design a lightweight intake side plate...",
    )

    context = st.text_area(
        "Additional context",
        key="context_draft",
        height=120,
        placeholder=(
            "Optional: existing CAD, manufacturing requirements, "
            "coordinate conventions, known interfaces..."
        ),
    )

    if st.button("Build formulation", type="primary"):
        if not request.strip():
            st.error("Enter a problem description first.")
        else:
            try:
                with st.spinner("Building and reviewing the formulation..."):
                    save_session(start_session(request, context.strip() or None))
                st.rerun()
            except Exception as error:
                model_error(error)


else:
    session = st.session_state.session

    top_left, top_right = st.columns([1, 1])

    with top_left:
        st.subheader("Current formulation")

        st.write(f"**Problem:** {session.spec.name}")
        st.write(f"**Type:** {session.spec.problem_kind}")

        if session.spec.physics:
            st.write(
                f"**Physics:** {session.spec.physics.family}"
            )

        if session.spec.geometry:
            st.write(
                f"**Geometry:** "
                f"{session.spec.geometry.description or 'not stated'}"
            )

        if session.spec.materials:
            names = [
                material.name or "unspecified"
                for material in session.spec.materials
            ]
            st.write(
                f"**Material:** {', '.join(names)}"
            )

        if session.spec.loads:
            st.write("**Loads:**")
            for load in session.spec.loads:
                magnitude = ""

                if load.magnitude:
                    magnitude = f" — {load.magnitude.value}"
                    if load.magnitude.unit:
                        magnitude += f" {load.magnitude.unit}"

                st.write(
                    f"- {load.name}{magnitude}"
                )

        if session.spec.optimization:
            st.write("**Objective:**")

            for objective in session.spec.optimization.objectives:
                st.write(
                    f"- {objective.sense or ''} "
                    f"{objective.quantity or 'unspecified'}"
                )

    with top_right:
        st.subheader("Geometry preview")

        preview = geometry_preview(session.spec)

        if preview is not None:
            st.pyplot(preview, width="stretch")
            import matplotlib.pyplot as plt
            plt.close(preview)
            st.caption(
                "Geometry from the current specification. Initial holes may change during optimization."
            )
        else:
            st.caption(
                "Not enough geometric information for a preview."
            )

        with st.expander("Problem structure"):
            st.graphviz_chart(
                problem_graph(session.spec),
                width="stretch",
            )


    st.divider()


    python_blockers = deterministic_blockers(session)
    if any(issue.key == "formulation_error" for issue in session.review.issues):
        if st.button("Retry formulation review"):
            with st.spinner("Retrying the review..."):
                save_session(retry_review(session))
            st.rerun()

    assessment = assess_spec(session.spec)
    if ready_to_try(session) and assessment.ready:
        st.success(
            "Ready to try: no known blocking formulation "
            "decisions remain."
        )

        st.subheader("Review and run")
        st.json(assessment.config.to_dict(), expanded=False)
        st.caption("Solver inputs use metres, newtons and pascals. The p-norm stress limit does not bound peak stress. Finishing the iteration budget does not establish convergence.")
        with st.expander("Run budget"):
            iterations = int(st.number_input("Optimization iterations", 1, 500, 2, 1))
            ranks = int(st.number_input("MPI processes", 1, 8, 1, 1))
        plan = run_plan(session, iterations, ranks)
        approved = st.checkbox("I approve these physical inputs and this run budget.", key="approve_" + plan["approval_hash"])
        running = bool(st.session_state.run_dir and run_status(Path(st.session_state.run_dir))["state"] in {"starting", "running"})
        if st.button("Run 3D L-bracket", disabled=not approved or running, type="primary"):
            try:
                st.session_state.run_dir = str(launch_run(session, approved_hash=plan["approval_hash"], max_iterations=iterations, ranks=ranks))
                st.rerun()
            except (ValueError, RuntimeError, OSError) as error:
                st.error(str(error))

    else:
        st.warning(
            "More engineering information is needed before trying the solver."
        )

        for blocker in python_blockers:
            st.write(f"- {blocker}")
        for issue in session.review.issues:
            if issue.blocking and issue.description not in python_blockers:
                st.write(f"- {issue.description}")


    questions = session.review.questions

    if questions:
        st.subheader("Questions for the engineer")

        answers = {}

        with st.form("clarification_form"):
            for question in questions:
                answer_key = f"round_{len(session.revisions)}_{question.key}"
                st.markdown(f"**{question.prompt}**")
                st.caption(question.why)

                if (
                    question.answer_type == "single_choice"
                    and question.options
                ):
                    choice = st.radio(
                        "Choose one",
                        question.options,
                        key=answer_key,
                        index=None,
                    )

                    other_text = ""

                    if any(
                        option.lower().startswith("other")
                        for option in question.options
                    ):
                        other_text = st.text_input(
                            "If you choose Other, describe it here",
                            key=f"{answer_key}_other",
                        )

                    if choice and choice.lower().startswith("other") and other_text:
                        answers[question.key] = other_text
                    else:
                        answers[question.key] = choice
                                        

                else:
                    answers[question.key] = st.text_input(
                        "Your answer",
                        key=answer_key,
                    )

            submitted = st.form_submit_button(
                "Update specification",
                type="primary",
            )

        if submitted:
            clean_answers = {
                key: value
                for key, value in answers.items()
                if value
            }

            if not clean_answers:
                st.error("Answer at least one question.")
            else:
                try:
                    with st.spinner("Updating the formulation..."):
                        save_session(answer_questions(session, clean_answers))
                    st.session_state.run_dir = None
                    st.rerun()
                except Exception as error:
                    model_error(error)


    with st.expander("Add engineering information or a correction"):
        supplement = st.text_area("Additional information", placeholder="Provide a clarified requirement or authoritative material data and its source.")
        if st.button("Apply additional information"):
            if supplement.strip():
                with st.spinner("Checking the added information..."):
                    save_session(answer_questions(session, {"additional_information": supplement.strip()}))
                st.session_state.run_dir = None
                st.rerun()
            else:
                st.error("Enter the information to add.")
    st.download_button("Download formulation and history", session.model_dump_json(indent=2), "formulation.json", "application/json")
    with st.expander("Request and question history"):
        st.write(session.original_request)
        st.write(session.context or "")
        for index, revision in enumerate(session.revisions, 1):
            st.markdown(f"**Round {index}**")
            st.json(revision.model_dump(mode="json"))

    @st.fragment(run_every="3s")
    def show_run(directory):
        run_dir = Path(directory)
        status = run_status(run_dir)
        last_state = st.session_state.get("last_run_state")
        st.session_state.last_run_state = status["state"]
        if last_state in {"starting", "running"} and status["state"] not in {"starting", "running"}:
            st.rerun()
        st.subheader("Run status")
        st.write(f"**{status['state'].capitalize()}** — {run_dir.name}")
        if status.get("reason"):
            st.warning(status["reason"])
        summary = run_dir / "solver" / "summary.json"
        if summary.exists():
            st.json(json.loads(summary.read_text()))
            st.download_button("Download run summary", summary.read_bytes(), "summary.json", "application/json")
        log = run_dir / "solver.log"
        if not log.exists():
            log = run_dir / "worker.log"
        if log.exists():
            with st.expander("Solver log", expanded=status["state"] == "failed"):
                st.code(log.read_text(errors="replace")[-12000:])
        st.caption(f"Saved inputs, approval, logs and fields: {run_dir}")

    if st.session_state.run_dir:
        show_run(st.session_state.run_dir)

    with st.expander("Developer details"):
        st.write(
            "**AI mode:**",
            "cached or live is shown per call below",
        )

        total_input = sum(
            call.input_tokens
            for call in session.usage
        )
        total_output = sum(
            call.output_tokens
            for call in session.usage
        )

        st.write(
            f"**Tokens:** {total_input:,} input / "
            f"{total_output:,} output"
        )

        for call in session.usage:
            mode = "CACHE HIT" if call.cache_hit else "LIVE"

            st.write(
                f"- {call.step}: "
                f"{call.input_tokens:,} in / "
                f"{call.output_tokens:,} out — "
                f"{mode}"
            )

        st.markdown("**Current structured specification**")
        st.json(
            session.spec.model_dump(
                exclude_none=True
            )
        )

        st.markdown("**Current review**")
        st.json(
            session.review.model_dump(
                exclude_none=True
            )
        )


    st.divider()

    if st.button("Start new problem"):
        st.session_state.session = None
        st.session_state.session_id = None
        st.session_state.run_dir = None
        st.session_state.request_draft = ""
        st.session_state.context_draft = ""
        st.rerun()
