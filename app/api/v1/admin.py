from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import List
from app.db.session import get_db
from app.api import deps
from app.db.models.question_model import Question
from app.db.models.test_model import VocationalTestResult
from pydantic import BaseModel

router = APIRouter()

class QuestionCreate(BaseModel):
    text: str
    category: str

class QuestionUpdate(BaseModel):
    text: str
    category: str
    is_active: bool

@router.get("/questions", response_model=List[dict])
async def list_questions(
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(deps.get_current_admin_user)
):
    result = await db.execute(select(Question))
    return [
        {"id": q.id, "text": q.text, "category": q.category, "is_active": q.is_active}
        for q in result.scalars().all()
    ]

@router.post("/questions")
async def create_question(
    question_in: QuestionCreate,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(deps.get_current_admin_user)
):
    db_question = Question(**question_in.model_dump())
    db.add(db_question)
    await db.commit()
    await db.refresh(db_question)
    return db_question

@router.delete("/questions/{question_id}")
async def delete_question(
    question_id: int,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(deps.get_current_admin_user)
):
    await db.execute(delete(Question).where(Question.id == question_id))
    await db.commit()
    return {"message": "Question deleted"}

@router.get("/results", response_model=List[dict])
async def view_all_results(
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(deps.get_current_admin_user)
):
    from app.db.models.user_model import User
    from sqlalchemy.orm import joinedload
    
    # Query with join to get user details
    query = select(VocationalTestResult).options(joinedload(VocationalTestResult.user))
    result = await db.execute(query)
    results = result.scalars().all()
    
    return [
        {
            "id": r.id, 
            "user_id": r.user_id, 
            "student_email": r.user.email if r.user else "Usuario eliminado",
            "student_name": r.user.full_name if r.user else "N/A",
            "recommendation": r.recommendation, 
            "scores": r.scores,
            "created_at": r.created_at
        }
        for r in results
    ]

from app.services.ml_service import ml_service

@router.get("/results/{result_id}/explanation")
async def explain_result(
    result_id: int,
    db: AsyncSession = Depends(get_db),
    admin: dict = Depends(deps.get_current_admin_user)
):
    from sqlalchemy.orm import joinedload
    result = await db.execute(
        select(VocationalTestResult)
        .options(joinedload(VocationalTestResult.user))
        .where(VocationalTestResult.id == result_id)
    )
    db_result = result.scalar_one_or_none()
    
    if not db_result:
        raise HTTPException(status_code=404, detail="Result not found")
    
    # Trace the path using the model
    explanation = ml_service.explain_prediction(db_result.scores)
    
    return {
        "result_id": result_id,
        "student_email": db_result.user.email if db_result.user else "N/A",
        "student_name": db_result.user.full_name if db_result.user else "N/A",
        "recommendation": db_result.recommendation,
        "scores": db_result.scores,
        **explanation
    }
