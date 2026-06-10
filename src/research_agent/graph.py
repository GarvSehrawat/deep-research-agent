from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from research_agent.config import settings
from research_agent.state import ResearchState
from research_agent.nodes.planner import planner_node
from research_agent.nodes.researcher import researcher_node
from research_agent.nodes.critic import critic_node
from research_agent.nodes.synthesizer import synthesizer_node
from typing import cast

def route_to_researchers(state) -> list[Send]:
    """Emit one Send per subtask → N parallel researcher branches."""
    return [
        Send("researcher", {"subtasks": [task], "findings": []})
        for task in state["subtasks"]
    ]

def route_after_critic(state) -> str:
    if state.get("_sufficient") or state["iterations"] >= settings.max_research_iterations:
        return "synthesizer"
    return "planner"

def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_node("critic", critic_node)
    g.add_edge(START, "planner")
    g.add_conditional_edges("planner", route_to_researchers, ["researcher"])
    g.add_edge("researcher", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["planner", "synthesizer"])
    g.add_edge("synthesizer", END)
    return g.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke(
        cast("ResearchState", {"query": "What are the trade-offs of RAG vs fine-tuning?"}),
        config={
            "run_name": "deep-research",
            "metadata": {
                "model_planner": settings.model_planner,
                "model_researcher": settings.model_researcher,
                "model_synthesizer": settings.model_synthesizer,
                "model_critic": settings.model_critic,
            },
        },
    )
    print(result["report"])