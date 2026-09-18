"""Terminal front end for the same pre-solve workflow used by Streamlit."""

from __future__ import annotations

from project.formulation.preview import issue_summary, spec_json
from project.formulation.session import approve_session, continue_session, start_session


def _answer_packet(session):
    answers: dict[str, str] = {}
    questions = session.critic_result.clarification_packet.questions
    print(f"\nCLARIFICATION PACKET ({len(questions)} items)")
    print("Answer what you can; press Enter to leave an item unresolved.\n")
    for index, question in enumerate(questions, start=1):
        print(f"{index}. {question.prompt}")
        print(f"   Why: {question.why_needed}")
        answer = input("   > ").strip()
        answers[question.id] = answer
    return answers


def main() -> None:
    print("Engineering Problem Formulation\n")
    context = input("Optional context (Enter to skip):\n> ").strip() or None
    problem = input("\nEngineering problem:\n> ").strip()

    session = start_session(problem, context)
    print("\nASSISTANT\n" + session.messages[-1].content)

    while session.status not in {"approved", "cancelled"}:
        if session.status == "ready_for_approval":
            print("\nCURRENT SPECIFICATION")
            print(spec_json(session))
            print("\n" + issue_summary(session))
            answer = input("\nApprove formulation? [y/N] or enter a correction:\n> ").strip()
            if answer.lower() in {"y", "yes", "approve", "approved"}:
                session = approve_session(session)
                print("\nASSISTANT\n" + session.messages[-1].content)
                break
            if not answer:
                continue
            session = continue_session(session, answer)
        else:
            packet = session.critic_result.clarification_packet.questions
            if packet:
                answers = _answer_packet(session)
                if any(value for value in answers.values()):
                    session = continue_session(session, structured_answers=answers)
                else:
                    freeform = input("\nNo packet answers supplied. Free-form clarification (Enter to retry):\n> ").strip()
                    if not freeform:
                        continue
                    session = continue_session(session, freeform)
            else:
                answer = input("\nYou:\n> ").strip()
                if not answer:
                    continue
                session = continue_session(session, answer)

        print("\nASSISTANT\n" + session.messages[-1].content)

    print("\nFinal status:", session.status)
    if session.status == "approved":
        print("\nAPPROVED SPECIFICATION")
        print(spec_json(session))


if __name__ == "__main__":
    main()
