from langgraph.types import Command
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from mind import build_graph

console = Console()


def main() -> None:
    graph = build_graph()
    thread_id = 0
    while True:
        question = input("\nEnter your request (or 'quit' to exit): ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        thread_id += 1
        config = {"configurable": {"thread_id": str(thread_id)}}
        with console.status("Thinking...", spinner="dots"):
            result = graph.invoke({"question": question}, config)
        while "__interrupt__" in result:
            clarifying_question = result["__interrupt__"][0].value
            console.print(
                Panel(clarifying_question, title="Clarifying question", border_style="cyan")
            )
            clarification = input("Your answer: ")
            with console.status("Thinking...", spinner="dots"):
                result = graph.invoke(Command(resume=clarification), config=config)
        console.print(Panel(result["goal"], title="Goal", border_style="green"))
        if result.get("plan"):
            console.print(Panel(Markdown(result["plan"]), title="Plan", border_style="blue"))
        console.print(Panel(Markdown(result["answer"]), title="Answer", border_style="magenta"))