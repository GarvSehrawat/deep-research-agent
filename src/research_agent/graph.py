from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from research_agent.config import settings
from research_agent.guardrails import RETRY_POLICY
from research_agent.state import ResearchState
from research_agent.nodes.planner import planner_node
from research_agent.nodes.researcher import researcher_node
from research_agent.nodes.critic import critic_node
from research_agent.nodes.synthesizer import synthesizer_node
from research_agent.nodes.guardrails import input_guard_node, output_guard_node
from typing import cast

def route_after_input_guard(state) -> str:
    """Input guardrail: refuse out-of-scope/unsafe queries before any spend."""
    return END if state.get("rejected") else "planner"

def route_to_researchers(state) -> list[Send]:
    """Emit one Send per subtask → N parallel researcher branches."""
    return [
        Send("researcher", {"subtasks": [task], "findings": []})
        for task in state["subtasks"]
    ]

def route_after_critic(state) -> str:
    """Stop the research loop when the findings are sufficient, the iteration cap
    is reached, OR the budget guard trips — whichever comes first. Any of these
    falls through to the synthesizer so we still ship a report from what we have."""
    budget_spent = state.get("llm_calls", 0) >= settings.max_llm_calls_per_run
    if state.get("_sufficient") or state["iterations"] >= settings.max_research_iterations or budget_spent:
        return "synthesizer"
    return "planner"

def route_after_output_guard(state) -> str:
    """Output guardrail: send a failing report back to be repaired, else deliver."""
    return "synthesizer" if state.get("_needs_repair") else END

def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("input_guard", input_guard_node, retry_policy=RETRY_POLICY)
    g.add_node("planner", planner_node, retry_policy=RETRY_POLICY)
    g.add_node("researcher", researcher_node, retry_policy=RETRY_POLICY)   # backoff handles 429s on fan-out
    g.add_node("critic", critic_node, retry_policy=RETRY_POLICY)
    g.add_node("synthesizer", synthesizer_node, retry_policy=RETRY_POLICY)
    g.add_node("output_guard", output_guard_node)   # no LLM call → no retry needed

    g.add_edge(START, "input_guard")
    g.add_conditional_edges("input_guard", route_after_input_guard, ["planner", END])
    g.add_conditional_edges("planner", route_to_researchers, ["researcher"])
    g.add_edge("researcher", "critic")
    g.add_conditional_edges("critic", route_after_critic, ["planner", "synthesizer"])
    g.add_edge("synthesizer", "output_guard")
    g.add_conditional_edges("output_guard", route_after_output_guard, ["synthesizer", END])
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
    if result.get("guardrail_flags"):
        print("\n⚠ guardrail flags:", result["guardrail_flags"])
    print(f"\nLLM calls this run: {result.get('llm_calls')}")