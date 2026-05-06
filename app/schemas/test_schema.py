from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class VocationalTestSubmission(BaseModel):
    # Dictionary of RIASEC scores or raw answers that map to RIASEC
    # For simplicity, we'll assume the frontend sends the calculated RIASEC scores (0-10)
    scores: Dict[str, int]

class VocationalTestResultResponse(BaseModel):
    id: int
    scores: Dict[str, int]
    recommendation: str
    details: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class CareerInfo(BaseModel):
    category: str
    description: str
    careers: List[str]
