from typing import TypedDict

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from mind.llm import complete_weak


class State(TypedDict):
    question: str
    goal: str
    clarifying_question: str
    clarification: str
    plan: str
    answer: str


def stage1_goal_definer(state: State) -> dict:
    print("=" * 60)
    print("[STAGE 1] goal definer")
    print("Original request:", state["question"])
    clarifying_question = complete_weak(
        [
            HumanMessage(
                content=(
                    f"Original request:\n{state['question']}\n\n"
                    "Ask exactly ONE clarifying question that would help "
                    "define the best goal for this request. "
                    "Just the question, nothing else."
                )
            )
        ]
    )
    print("-" * 60)
    print("Clarifying question:", clarifying_question)
    clarification = interrupt(clarifying_question)
    print("User clarification:", clarification)
    goal = complete_weak(
        [
            HumanMessage(
                content=(
                    f"Original request:\n{state['question']}\n\n"
                    f"User's clarification:\n{clarification}\n\n"
                    "Define a clear, concise goal for this request."
                )
            )
        ]
    )
    print("-" * 60)
    print("Goal:", goal)
    return {
        "goal": goal,
        "clarifying_question": clarifying_question,
        "clarification": clarification,
    }


def stage2_executor(state: State) -> dict:
    print("=" * 60)
    print("[STAGE 2] executor")
    print("Input question:", state["question"])
    print("Goal:", state["goal"])
    print("Clarification:", state["clarification"])
    plan = complete_weak(
        [
            HumanMessage(
                content=(
                    f"Original request:\n{state['question']}\n\n"
                    f"Goal:\n{state['goal']}\n\n"
                    f"Clarification from user:\n{state['clarification']}\n\n"
                    "What is the absolute best way to solve this goal? "
                    "Give a concise, concrete plan. Nothing else."
                )
            )
        ]
    )
    print("-" * 60)
    print("Plan:", plan)
    answer = complete_weak(
        [
            HumanMessage(
                content=(
                    f"Original request:\n{state['question']}\n\n"
                    f"Goal:\n{state['goal']}\n\n"
                    f"Clarification from user:\n{state['clarification']}\n\n"
                    f"Plan:\n{plan}\n\n"
                    "Execute the plan to achieve the goal."
                )
            )
        ]
    )
    print("-" * 60)
    print("Answer:", answer)
    return {"plan": plan, "answer": answer}


def build_graph():
    builder = StateGraph(State)
    builder.add_node("stage1_goal_definer", stage1_goal_definer)
    builder.add_node("stage2_executor", stage2_executor)
    builder.add_edge(START, "stage1_goal_definer")
    builder.add_edge("stage1_goal_definer", "stage2_executor")
    builder.add_edge("stage2_executor", END)
    return builder.compile(checkpointer=MemorySaver())


def main() -> None:
    graph = build_graph()
    config = {"configurable": {"thread_id": "1"}}
    result = graph.invoke({"question": "What is the meaning of life?"}, config)
    question = result["__interrupt__"][0].value
    clarification = input("Your answer: ")
    result = graph.invoke(Command(resume=clarification), config=config)
    print("=" * 60)
    print("FINAL STATE")
    for key in ("question", "goal", "clarifying_question", "clarification", "plan", "answer"):
        print(f"{key}: {result[key]}")