from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class IncidentRecord(BaseModel):
    incident_id: str
    incident_state: str = ""
    impact: int = 2
    urgency: int = 2
    priority: str = "P3"
    description: str = ""
    assigned_to: str = ""
    resolution_notes: str = ""
    resolved_at: Optional[datetime] = None
    index_text: str = ""


class RetrievedIncident(BaseModel):
    incident_id: str
    description: str
    resolution_notes: str
    priority: str
    impact: int
    assigned_to: str
    incident_state: str
    resolved_at: Optional[datetime] = None
    rrf_score: float = 0.0
    combined_score: float = 0.0
    confidence: str = "MEDIUM"
    recency_days: Optional[int] = None
