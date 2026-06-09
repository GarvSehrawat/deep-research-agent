from typing import Annotated, TypedDict
from operator import add

from pydantic import BaseModel

class Finding(BaseModel):
    claim: str
    source_url: str
    source_title: str

class ResearchState(TypedDict):
    query: str                                # set at input
    subtasks: list[str]                       # written by planner
    findings: Annotated[list[Finding], add]   # ← reducer: ACCUMULATE
    report: str                               # written by synthesizer
    iterations: int                           # bumped each critic pass
    gaps: list[str]                           # critic's notes for the next round