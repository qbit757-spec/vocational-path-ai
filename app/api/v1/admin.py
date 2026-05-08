from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import List, Optional
from pydantic import BaseModel
from app.db.session import get_db
from app.api import deps
from app.db.models.question_model import Question
from app.db.models.test_model import VocationalTestResult
from app.services.ml_service import ml_service

router = APIRouter()

class TrainRequest(BaseModel):
    filenames: Optional[List[str]] = None

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

@router.post("/datasets/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    admin: dict = Depends(deps.get_current_admin_user)
):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files allowed")
    
    content = await file.read()
    path = ml_service.save_dataset(file.filename, content)
    return {"filename": file.filename, "path": path}

@router.get("/datasets")
async def list_datasets(
    admin: dict = Depends(deps.get_current_admin_user)
):
    return ml_service.list_datasets()

@router.post("/train")
async def train_model_endpoint(
    request: TrainRequest = Body(None),
    admin: dict = Depends(deps.get_current_admin_user)
):
    try:
        filenames = request.filenames if request else None
        stats = await ml_service.train_from_files(filenames)
        return {"message": "Model trained successfully", "stats": stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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
    
    formatted_results = []
    for r in results:
        riasec_scores = {"R": 0, "I": 0, "A": 0, "S": 0, "E": 0, "C": 0}
        if r.scores and isinstance(r.scores, dict):
            for key, val in r.scores.items():
                if key and key[0] in riasec_scores and key[1:].isdigit():
                    try: riasec_scores[key[0]] += int(val)
                    except: pass
                    
        prob = 0.0
        explanation = ml_service.explain_prediction(r.scores or {})
        if explanation and "insights" in explanation:
            prob = float(explanation["insights"].get("confidence", 0.0))
            
        formatted_results.append({
            "id": r.id,
            "student_name": r.user.full_name if r.user else "N/A",
            "student_email": r.user.email if r.user else "Usuario eliminado",
            "created_at": r.created_at,
            "prediction": r.recommendation,
            "probability": prob,
            "riasec_scores": riasec_scores
        })
    return formatted_results

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
    
    if not explanation:
        raise HTTPException(status_code=400, detail="El modelo no ha sido entrenado aún. Por favor, entrena el modelo primero.")
    
    return {
        "result_id": result_id,
        "student_email": db_result.user.email if db_result.user else "N/A",
        "student_name": db_result.user.full_name if db_result.user else "N/A",
        "recommendation": db_result.recommendation,
        "scores": db_result.scores,
        **explanation
    }
