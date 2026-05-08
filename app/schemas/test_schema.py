from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class Answer(BaseModel):
    question_id: int
    value: int

class VocationalTestSubmission(BaseModel):
    answers: List[Answer]

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
