# LangGaph에서 state 공유
from typing import List
from pydantic import BaseModel

class AgentState(BaseModel):
    query: str
    candidates: List[str] = []
    evidence: List[str] = []
    ranked_results: List[str] = []
