from pydantic import BaseModel, Field
from research_agent.models import get_model

class Plan(BaseModel):
    subtasks: list[str] = Field(description="3-5 focused, non-overlapping research questions")

def planner_node(state):
    model = get_model("planner").with_structured_output(Plan)

    gaps = state.get("gaps")          # ← empty on first pass, populated after a critic loop
    if gaps:
        prompt = (
            f"Original query: {state['query']}\n\n"
            f"A first research pass left these gaps:\n{gaps}\n\n"
            "Produce focused subtasks that close ONLY these gaps."
        )
    else:
        prompt = f"Break this into research subtasks:\n{state['query']}"

    plan = model.invoke(prompt)
    assert isinstance(plan, Plan)
    return {"subtasks": plan.subtasks}