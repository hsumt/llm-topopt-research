from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="Engineering Formulation Assistant",
    page_icon="⚙️",
    layout="wide",
)


def initialize_state() -> None:
    defaults = {
        "messages": [],
        "stage": "awaiting_problem",
        "parser_result": None,
        "problem_spec": None,
        "formulation_session": None,
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


st.title("Engineering Problem Formulation")
st.caption(
    "Translate an engineering request into a reviewed, "
    "solver-independent problem specification."
)


main_col, side_col = st.columns([2.2, 1])


with main_col:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    user_input = st.chat_input(
        "Describe the engineering problem or answer the current question..."
    )

    if user_input:
        st.session_state.messages.append(
            {
                "role": "user",
                "content": user_input,
            }
        )

        if st.session_state.stage == "awaiting_problem":
            # Later:
            # parser_result, usage = parse_problem(user_input)
            #
            # st.session_state.parser_result = parser_result
            # st.session_state.problem_spec = parser_result.spec
            # critic_result = review_formulation(...)
            #
            # assistant_text = ...
            # st.session_state.stage = "awaiting_clarification"

            assistant_text = (
                "Parser integration will go here. "
                "The initial engineering request has been received."
            )

        else:
            # Later:
            # resolution = resolve_response(...)
            # patched_spec = apply_patch(...)
            # critic_result = review_formulation(...)
            assistant_text = (
                "Formulation-dialogue handling will go here."
            )

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": assistant_text,
            }
        )

        st.rerun()


with side_col:
    st.subheader("Current State")

    st.write("**Stage:**", st.session_state.stage)

    tab_spec, tab_prov, tab_issues = st.tabs(
        ["Specification", "Provenance", "Issues"]
    )

    with tab_spec:
        if st.session_state.problem_spec is None:
            st.info("No problem has been parsed yet.")
        else:
            st.json(
                st.session_state.problem_spec.model_dump(
                    exclude_none=True
                )
            )

    with tab_prov:
        result = st.session_state.parser_result

        if result is None:
            st.info("No provenance available yet.")
        else:
            st.json(
                [
                    item.model_dump()
                    for item in result.field_provenance
                ]
            )

    with tab_issues:
        result = st.session_state.parser_result

        if result is None:
            st.info("No formulation issues available yet.")
        else:
            st.write("### Unresolved")
            for item in result.unresolved_items:
                st.warning(item.issue)

            st.write("### Contradictions")
            for item in result.contradictions:
                st.error(item.description)


with st.sidebar:
    st.header("Session")

    if st.button("New problem"):
        st.session_state.clear()
        st.rerun()