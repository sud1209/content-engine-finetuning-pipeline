"""
Pydantic models for the FastAPI scoring endpoints.
"""

from pydantic import BaseModel, Field


class ScoreRequest(BaseModel):
    content: str = Field(min_length=10, max_length=1000)


class ScoreResponse(BaseModel):
    hook_strength: int
    tone_compliance: int
    x_algorithm_optimization: int
    data_specificity: int
    pillar_alignment: int
    cta_quality: int
    never_list_violation: bool
    composite_score: float
    reasoning: str
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model: str
    version: str = "1.0.0"
