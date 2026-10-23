"""The questions for one round, answered in place. Each shows why it is asked (who raised it, the issue it
settles, the spec field and what is stated there, the solver input it feeds) and lets the engineer ask the
model to explain or reword it; the reply is explanation only and never an answer."""
import re

import streamlit as st

from project.ui import api
from project.ui.components.issues_view import NEEDS_DECISION, UNSUPPORTED, mark
from project.ui.components.spec_view import md

_PATH = re.compile(r"[A-Za-z_]\w*(?:\.\w+|\[\d+\])*")   # a well-formed spec path; anything else is not shown as code
_KIND_WORDS = {"decision": "Needs your decision", "data": "Missing data", "unsupported": "Not supported"}
_MODES = {"explain": "Explaining the question", "reword": "Rewording the question", "ask": "Answering your question"}


def answer_key(fid: str, round_index: int, question_key: str) -> str:
    """Session-state key of one answer box; the walkthrough fills boxes through it."""
    return f"answer_{fid}_{round_index}_{question_key}"


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


def _line(text: str) -> str:
    """Model or engineer text as one line, so line-level markdown (headings, lists, rules) cannot form."""
    return " ".join(str(text).split())


def _reasoning(question, context: dict | None) -> None:
    """Why the question is asked, from what the system recorded; model text is labelled as such."""
    context = context or {"origin": "code" if question.key.startswith("solver_") else "model", "issues": []}
    with st.container(border=True):
        if context["origin"] == "code":
            cite = context.get("cite") or {}
            where = (f"[{cite['label']}]({cite['open']})" if cite.get("open") else f"`{cite['label']}`") if cite else ""
            st.markdown(f":material/rule: **Why this is asked** · raised by the code check"
                        + (f" (rule {where})" if where else ""))
        else:
            st.markdown(":material/smart_toy: **Why this is asked** · raised by the reviewer model")
        lines = []
        for issue in context["issues"]:
            role = UNSUPPORTED if issue["kind"] == "unsupported" else NEEDS_DECISION
            facts = []
            if issue.get("field"):
                field = issue["field"]
                facts.append(f"spec field `{field}`" if _PATH.fullmatch(field) else "spec field not identified")
            facts.append(f"stated now: {md(issue['stated'])}" if issue.get("stated") else "stated now: nothing")
            if issue.get("inputs"):
                facts.append("feeds solver input " + ", ".join(f"`{name}`" for name in issue["inputs"]))
            lines.append(f"- {mark(role)} {_KIND_WORDS.get(issue['kind'], issue['kind'])}: {md(_line(issue['description']))}  \n"
                         f"  :small[{' · '.join(facts)}]")
        if lines:
            st.markdown("\n".join(lines))
        described = {issue["description"] for issue in context["issues"]}
        if question.why and question.why not in described:
            st.caption(("Reviewer model's reasoning (model text, not checked by code): " if context["origin"] == "model"
                        else "") + md(_line(question.why)))


def _ask(fid: str, question, clarifications: list) -> None:
    """Ask the model about the question; show what it said before (this round), newest first."""
    with st.popover("Ask about this question", icon=":material/help_center:"):
        st.caption("The model explains; it does not answer for you or change the specification. "
                   "Your answer below is what counts.")
        query = st.text_input("Your question about it", key=f"clarify_query_{fid}_{question.key}",
                              placeholder="For example: what does 'restate' mean here?")
        columns = st.columns(3)
        clicked = [(mode, column.button(label, key=f"clarify_{mode}_{fid}_{question.key}"))
                   for mode, column, label in zip(("explain", "reword", "ask"), columns, ("Explain it", "Reword it", "Ask"))]
        pressed = next((mode for mode, hit in clicked if hit), None)
        if pressed:
            with st.spinner(_MODES[pressed] + "…"):
                try:
                    api.clarify(fid, question.key, pressed, query if pressed == "ask" else "")
                except (ValueError, RuntimeError) as error:
                    st.error(str(error))
                else:
                    st.rerun()
    for record in reversed(clarifications[-3:]):
        with st.container(border=True):
            asked = {"explain": "You asked for an explanation", "reword": "You asked for a rewording"}.get(
                record.mode, f"You asked: {md(_line(record.query))}")
            st.caption(f":material/smart_toy: {asked} · the model's reply is explanation only, not an answer")
            if record.failure:
                st.error(md(_line(record.failure)))
                continue
            if record.rewording:
                st.markdown(f"**Reworded:** {md(_line(record.rewording))}")
            if record.explanation:
                st.markdown(md(_line(record.explanation)))


def _question(fid: str, round_index: int, index: int, question, context: dict | None = None,
              clarifications: list | None = None) -> str | None:
    key = answer_key(fid, round_index, question.key)
    with st.container(border=True):
        st.markdown(f"**{index}. {md(question.prompt)}**")
        _reasoning(question, context)
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
        _ask(fid, question, clarifications or [])
    return answer or None


def render(fid: str, questions: list, round_index: int, context: dict | None = None,
           clarifications: list | None = None) -> dict[str, str] | None:
    """context: api.question_context (why each is asked); clarifications: the session's, any round."""
    context, clarifications = context or {}, clarifications or []
    answers = {}
    for index, question in enumerate(questions, 1):
        mine = [c for c in clarifications if c.round == round_index and c.question_key == question.key]
        answer = _question(fid, round_index, index, question, context.get(question.key), mine)
        if answer:
            answers[question.key] = answer
    st.caption("Answer what you can. The reviewer then checks the updated specification "
               "and asks again about anything still missing.")
    if st.button("Submit answers", type="primary", key=f"submit_{fid}_{round_index}"):
        return answers
    return None
