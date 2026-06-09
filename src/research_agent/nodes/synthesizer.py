from pydantic import BaseModel

from research_agent.models import get_model

class CitedReport(BaseModel):
    summary: str
    sections: list[str]
    citations: list[str]   # the source URLs actually used

def synthesizer_node(state):
    model = get_model("synthesizer").with_structured_output(CitedReport)
    report = model.invoke(
        "Write a report. Every claim must reference one of these sources "
        f"by URL.\n\nFindings:\n{state['findings']}"
    )
    if not isinstance(report, CitedReport):
        raise ValueError(f"Expected CitedReport, got {type(report)}")
    return {"report": report.model_dump()}
