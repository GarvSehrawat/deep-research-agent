from research_agent.models import get_model

def researcher_node(state):
    model = get_model("researcher")
    task = state["subtasks"][0]
    answer = model.invoke(f"Research this and report findings:\n{task}")
    return {"findings": [answer.content]}   # list → matches the reducer