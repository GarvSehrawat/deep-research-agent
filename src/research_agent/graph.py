from langgraph.graph import StateGraph, START, END
from research_agent.state import ResearchState
from research_agent.nodes.planner import planner_node
from research_agent.nodes.researcher import researcher_node
from research_agent.nodes.synthesizer import synthesizer_node
from typing import cast


def build_graph():
    g = StateGraph(ResearchState)
    g.add_node("planner", planner_node)
    g.add_node("researcher", researcher_node)
    g.add_node("synthesizer", synthesizer_node)
    g.add_edge(START, "planner")
    g.add_edge("planner", "researcher")
    g.add_edge("researcher", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


if __name__ == "__main__":
    app = build_graph()
    result = app.invoke(cast("ResearchState", {"query": "What are the trade-offs of RAG vs fine-tuning?"}))
    print(result["report"])