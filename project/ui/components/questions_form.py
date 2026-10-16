"""The reviewer's questions for one round, answered in place."""
import streamlit as st

from project.ui.components.spec_view import md


def _other(options: list[str]) -> str | None:
    return next((o for o in options if o.strip().lower().startswith("other")), None)


def _example_target(question, key: str) -> dict[str, str] | None:
    """Widget values that put the example answer in place, or None when it cannot be."""
    example = question.example_answer
    if question.answer_type != "single_choice" or not question.options:
        return {key: example}
    match = next((o for o in question.options if o.strip().lower() == example.strip().lower()), None)
    if match is not None:
        return {key: match}
    other = _other(question.options)
    return {key: other, f"{key}_other": example} if other else None


def _use_example(values: dict[str, str]) -> None:
    st.session_state.update(values)


def _question(fid: str, round_index: int, index: int, question) -> str | None:
    key = f"answer_{fid}_{round_index}_{question.key}"
    with st.container(border=True):
        st.markdown(f"**{index}. {md(question.prompt)}**")
        if question.why:
            st.caption(md(question.why))
        if question.answer_type == "single_choice" and question.options:
            choice = st.radio("Your answer", question.options, index=None, key=key, format_func=md,
                              label_visibility="collapsed")
            other = _other(question.options)
            if choice is not None and choice == other:
                text = st.text_input("Describe your answer", key=f"{key}_other")
                answer = text.strip()
            else:
                answer = choice or ""
        else:
            answer = st.text_area("Your answer", key=key, height=90, label_visibility="collapsed",
                                  placeholder="Type your answer").strip()
        if question.example_answer:
            target = _example_target(question, key)
            with st.container(border=True):
                st.caption(f"Example answer: {md(question.example_answer)}")
                if target is not None:
                    st.button("Use this example", key=f"example_{key}", on_click=_use_example, args=(target,))
    return answer or None


def render(fid: str, questions: list, round_index: int) -> dict[str, str] | None:
    answers = {}
    for index, question in enumerate(questions, 1):
        answer = _question(fid, round_index, index, question)
        if answer:
            answers[question.key] = answer
    st.caption("Answer what you can. The reviewer then checks the updated specification "
               "and asks again about anything still missing.")
    if st.button("Submit answers", type="primary", key=f"submit_{fid}_{round_index}"):
        return answers
    return None
