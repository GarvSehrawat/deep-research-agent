from typing import Annotated, TypedDict
from operator import add


class ResearchState(TypedDict):
    query: str                              # set at input
    subtasks: list[str]                     # written by planner
    findings: Annotated[list[str], add]     # ← reducer: ACCUMULATE
    report: str                             # written by synthesizer