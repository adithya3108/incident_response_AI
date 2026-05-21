from typing import Optional
from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    priority: Optional[list[str]] = None
    impact: Optional[list[int]] = None
    incident_state: Optional[list[str]] = None


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: Optional[MetadataFilter] = None


class BulkSearchRequest(BaseModel):
    queries: list[str] = Field(..., min_length=1, max_length=50)
    top_k: int = Field(default=3, ge=1, le=10)
    filters: Optional[MetadataFilter] = None


class TriageRequest(BaseModel):
    description: str = Field(..., min_length=3, max_length=2000)
    impact: Optional[int] = Field(default=None, ge=1, le=3)
    urgency: Optional[int] = Field(default=None, ge=1, le=3)


class EscalateRequest(BaseModel):
    incident_id: str
    current_tier: str = Field(default="L1", pattern="^(L1|L2|L3)$")
    description: str = Field(..., min_length=3, max_length=2000)
    resolution_attempts: list[str] = Field(default_factory=list)


class AnalyzeRequest(BaseModel):
    incident_ids: list[str] = Field(..., min_length=1, max_length=20)


class FeedbackRequest(BaseModel):
    incident_id: str
    query: str
    helpful: bool
    resolution_applied: bool = False
    comments: str = ""
