from project.examples import EXAMPLES
import streamlit as st

from project.drawing import (
    geometry_preview,
    problem_graph,
)
from project.workflow import (
    answer_questions,
    deterministic_blockers,
    ready_to_try,
    start_session,
)


st.set_page_config(
    page_title="Engineering Formulation",
    page_icon="⚙️",
    layout="wide",
)


if "session" not in st.session_state:
    st.session_state.session = None

if "request_draft" not in st.session_state:
    st.session_state.request_draft = ""

if "context_draft" not in st.session_state:
    st.session_state.context_draft = ""


st.title("Engineering Problem Formulation")
st.caption(
    "Turn an incomplete engineering request into a reviewable problem "
    "before attempting a solver."
)

with st.sidebar:
    st.subheader("Demo problems")

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

            st.rerun()

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
            with st.spinner("Building and reviewing the formulation..."):
                st.session_state.session = start_session(
                    request,
                    context.strip() or None,
                )

            st.rerun()


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
            st.pyplot(preview, use_container_width=True)
            st.caption(
                "CAD-governed geometry not yet positioned: "
                "mounting holes, intake pivot, bumper clearance."
            )
        else:
            st.caption(
                "Not enough geometric information for a preview."
            )

        with st.expander("Problem structure"):
            st.graphviz_chart(
                problem_graph(session.spec),
                use_container_width=True,
            )


    st.divider()


    python_blockers = deterministic_blockers(session)

    if ready_to_try(session):
        st.success(
            "Ready to try: no known blocking formulation "
            "decisions remain."
        )

        st.button(
            "Try solver",
            disabled=True,
            help="Solver handoff is not connected in this refactor yet.",
        )

    else:
        st.warning(
            "More engineering information is needed before trying the solver."
        )

        for blocker in python_blockers:
            st.write(f"- {blocker}")


    questions = session.review.questions

    if questions:
        st.subheader("Questions for the engineer")

        answers = {}

        with st.form("clarification_form"):
            for question in questions:
                st.markdown(f"**{question.prompt}**")
                st.caption(question.why)

                if (
                    question.answer_type == "single_choice"
                    and question.options
                ):
                    choice = st.radio(
                        "Choose one",
                        question.options,
                        key=question.key,
                        index=None,
                    )

                    other_text = ""

                    if any(
                        option.lower().startswith("other")
                        for option in question.options
                    ):
                        other_text = st.text_input(
                            "If you choose Other, describe it here",
                            key=f"{question.key}_other",
                        )

                    if choice and choice.lower().startswith("other") and other_text:
                        answers[question.key] = other_text
                    else:
                        answers[question.key] = choice
                                        

                else:
                    answers[question.key] = st.text_input(
                        "Your answer",
                        key=question.key,
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
                with st.spinner("Updating the formulation..."):
                    st.session_state.session = answer_questions(
                        session,
                        clean_answers,
                    )

                st.rerun()


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
        st.session_state.request_draft = ""
        st.session_state.context_draft = ""
        st.rerun()