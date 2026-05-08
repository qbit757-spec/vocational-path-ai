from pydantic import BaseModel
from typing import Dict, List, Optional
from datetime import datetime

class Answer(BaseModel):
    question_id: int
    value: int

class VocationalTestSubmission(BaseModel):
    answers: List[Answer]
    age: Optional[int] = 18
    gender: Optional[int] = 1 # 1=Male, 2=Female, 3=Other
    education: Optional[int] = 2 # 1=Less than high school, 2=High school, 3=University...

class VocationalTestResultResponse(BaseModel):
    id: int
    scores: Dict[str, int]
    riasec_percentages: Dict[str, int] = {}
    recommendation: str
    details: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True

class CareerInfo(BaseModel):
    category: str
    description: str
    careers: List[str]
