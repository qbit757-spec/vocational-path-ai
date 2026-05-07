from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List
from app.db.session import get_db
from app.api import deps
from app.db.models.user_model import User
from app.db.models.test_model import VocationalTestResult
from app.schemas.test_schema import VocationalTestSubmission, VocationalTestResultResponse
from app.ml.predictor import predictor

from app.db.models.question_model import Question

router = APIRouter()

@router.get("/questions", response_model=List[dict])
async def get_test_questions(
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Question).where(Question.is_active == True))
    questions = result.scalars().all()
    return [
        {"id": q.id, "text": q.text, "category": q.category}
        for q in questions
    ]

# Justification descriptions based on RIASEC categories
CAREER_INFO = {
    'Ingeniería / Tecnología': {
        'description': 'Muestras un fuerte interés por el funcionamiento de las cosas y la resolución de problemas técnicos.',
        'careers': ['Ingeniería Civil', 'Ingeniería de Sistemas', 'Ingeniería Mecánica', 'Arquitectura']
    },
    'Ciencias de la Salud': {
        'description': 'Tienes una inclinación natural hacia el cuidado de los demás y la investigación biológica.',
        'careers': ['Medicina', 'Enfermería', 'Psicología', 'Nutrición']
    },
    'Artes y Diseño': {
        'description': 'Tu perfil es altamente creativo y valoras la expresión estética.',
        'careers': ['Diseño Gráfico', 'Artes Escénicas', 'Arquitectura', 'Comunicación Audiovisual']
    },
    'Ciencias Sociales / Educación': {
        'description': 'Te interesa el bienestar social, la enseñanza y el entendimiento del comportamiento humano.',
        'careers': ['Educación', 'Sociología', 'Derecho', 'Trabajo Social']
    },
    'Negocios / Derecho': {
        'description': 'Posees habilidades persuasivas y te interesa la gestión de proyectos y el marco legal.',
        'careers': ['Administración de Empresas', 'Derecho', 'Marketing', 'Economía']
    },
    'Administración / Contabilidad': {
        'description': 'Eres organizado, detallista y disfrutas trabajando con datos y procesos estructurados.',
        'careers': ['Contabilidad', 'Finanzas', 'Administración Pública', 'Ciencia de Datos']
    }
}

@router.post("/submit", response_model=VocationalTestResultResponse)
async def submit_test(
    submission: VocationalTestSubmission,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        # 1. Use ML model to predict category
        recommendation = predictor.predict(submission.scores)
        
        # 2. Get justification and careers
        info = CAREER_INFO.get(recommendation, {'description': '', 'careers': []})
        details = f"{info['description']} Carreras sugeridas: {', '.join(info['careers'])}"
        
        # 3. Save result to DB
        db_result = VocationalTestResult(
            user_id=current_user.id,
            scores=submission.scores,
            recommendation=recommendation,
            details=details
        )
        db.add(db_result)
        await db.commit()
        await db.refresh(db_result)
        
        return db_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[VocationalTestResultResponse])
async def get_history(
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    result = await db.execute(
        select(VocationalTestResult)
        .where(VocationalTestResult.user_id == current_user.id)
        .order_by(VocationalTestResult.created_at.desc())
    )
    return result.scalars().all()
