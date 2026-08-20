from typing import Literal, TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel

from mind.llm import complete_weak, complete_weak_structured


class State(TypedDict):
    question: str
    goal: str
    clarifying_question: str
    clarification: str
    history: list
    plan: str
    answer: str


class InterviewDecision(BaseModel):
    decision: Literal["question", "goal"]
    text: str


def _format_history(history: list) -> str:
    if not history:
        return "No clarifications yet."
    return "\n".join(f"Q: {q}\nA: {a}" for q, a in history)


def ask_question(state: State) -> dict:
    try:
        result = complete_weak_structured(
            [
                HumanMessage(
                    content=(
                        f"Original request:\n{state['question']}\n\n"
                        f"Interview so far:\n{_format_history(state.get('history'))}\n\n"
                        "You are an interviewer who refuses to start building until "
                        "every ambiguity is resolved.\n"
                        "Decide whether the goal is now fully unambiguous:\n"
                        "- If more clarification is needed, set decision to 'question' "
                        "and put one precise clarifying question in 'text'.\n"
                        "- If the goal is now unambiguous, set decision to 'goal' and "
                        "put the clear, concise goal in 'text'."
                    )
                )
            ],
            InterviewDecision.model_json_schema(),
        )
        decision = InterviewDecision.model_validate(result)
    except Exception:
        return {"clarifying_question": "Could you clarify what you need further?", "goal": ""}

    text = decision.text.strip()
    if decision.decision == "goal":
        return {"goal": text}
    return {"clarifying_question": text, "goal": ""}


def get_answer(state: State) -> dict:
    clarification = interrupt(state["clarifying_question"])
    history = list(state.get("history", []))
    history.append((state["clarifying_question"], clarification))
    return {"clarification": clarification, "history": history}


def should_continue(state: State) -> str:
    return "done" if state.get("goal") else "ask_question"


def stage2_executor(state: State) -> dict:
    plan = complete_weak(
        [
            HumanMessage(
                content=(
                    f"Original request:\n{state['question']}\n\n"
                    f"Goal:\n{state['goal']}\n\n"
                    f"Interview so far:\n{_format_history(state.get('history'))}\n\n"
                    "What is the absolute best way to solve this goal? "
                    "Give a concise, concrete plan. Nothing else."
                )
            )
        ]
    )
    answer = complete_weak(
        [
            HumanMessage(
                content=(
                    f"Original request:\n{state['question']}\n\n"
                    f"Goal:\n{state['goal']}\n\n"
                    f"Interview so far:\n{_format_history(state.get('history'))}\n\n"
                    f"Plan:\n{plan}\n\n"
                    "Execute the plan to achieve the goal."
                )
            )
        ]
    )
    return {"plan": plan, "answer": answer}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("ask_question", ask_question)
    builder.add_node("get_answer", get_answer)
    builder.add_node("stage2_executor", stage2_executor)
    builder.add_edge(START, "ask_question")
    builder.add_conditional_edges(
        "ask_question",
        should_continue,
        {"ask_question": "get_answer", "done": "stage2_executor"},
    )
    builder.add_edge("get_answer", "ask_question")
    builder.add_edge("stage2_executor", END)
    return builder.compile(checkpointer=MemorySaver())