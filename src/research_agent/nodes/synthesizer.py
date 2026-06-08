from research_agent.models import get_model

def synthesizer_node(state):
    model = get_model("synthesizer")
    joined = "\n\n".join(state["findings"])
    report = model.invoke(f"Write a short report from these findings:\n{joined}")
    return {"report": report.content}