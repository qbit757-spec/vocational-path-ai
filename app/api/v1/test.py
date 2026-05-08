from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import List, Dict
from app.db.session import get_db
from app.api import deps
from app.db.models.user_model import User
from app.db.models.test_model import VocationalTestResult
from app.schemas.test_schema import VocationalTestSubmission, VocationalTestResultResponse
from app.services.ml_service import ml_service
from app.db.models.question_model import Question

router = APIRouter()

@router.get("/questions", response_model=List[dict])
async def get_test_questions(db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Question).where(Question.is_active == True))
    questions = result.scalars().all()
    return [{"id": q.id, "text": q.text, "category": q.category} for q in questions]

# CATEGORIAS ACTUALIZADAS (5 CATEGORIAS SOLIDAS)
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
        'careers': ['Diseño Gráfico', 'Artes Escénicas', 'Diseño de Modas', 'Comunicación Audiovisual']
    },
    'Ciencias Sociales / Educación': {
        'description': 'Te interesa el bienestar social, la enseñanza y el entendimiento del comportamiento humano.',
        'careers': ['Educación', 'Sociología', 'Derecho', 'Trabajo Social']
    },
    'Negocios, Gestión y Derecho': {
        'description': 'Posees habilidades persuasivas, te interesa la gestión estratégica y el marco legal.',
        'careers': ['Administración', 'Derecho', 'Contabilidad', 'Marketing', 'Economía']
    }
}

@router.post("/submit", response_model=VocationalTestResultResponse)
async def submit_test(
    submission: VocationalTestSubmission,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        # 1. Mapear respuestas a formato 48-item (R1, R2... C8)
        # Obtenemos las categorias de las preguntas para saber cual es cual
        q_result = await db.execute(select(Question))
        q_map = {q.id: q.category for q in q_result.scalars().all()}
        
        # Diccionario para contar cuantas preguntas de cada tipo llevamos
        cat_counts = {"Realista": 0, "Investigativo": 0, "Artístico": 0, "Social": 0, "Emprendedor": 0, "Convencional": 0}
        cat_to_letter = {"Realista": "R", "Investigativo": "I", "Artístico": "A", "Social": "S", "Emprendedor": "E", "Convencional": "C"}
        
        raw_scores = {
            "age": submission.age,
            "gender": submission.gender,
            "education": submission.education
        }
        for ans in submission.answers:
            cat_name = q_map.get(ans.question_id)
            if cat_name in cat_counts:
                cat_counts[cat_name] += 1
                letter = cat_to_letter[cat_name]
                # Guardamos como R1, R2... etc para que la IA lo entienda
                raw_scores[f"{letter}{cat_counts[cat_name]}"] = ans.value
        
        # 2. Usar ML Service para predecir (XGBoost)
        explanation = ml_service.explain_prediction(raw_scores)
        if not explanation:
            raise HTTPException(status_code=500, detail="Modelo no entrenado")
            
        recommendation = str(explanation["decision_path"][-1]["prediction"])
        
        # 3. Obtener info de carrera
        info = CAREER_INFO.get(recommendation, {'description': '', 'careers': []})
        details = f"{info['description']} Carreras sugeridas: {', '.join(info['careers'])}"
        
        # 4. Guardar resultado
        db_result = VocationalTestResult(
            user_id=current_user.id,
            scores=raw_scores,
            recommendation=recommendation,
            details=details
        )
        db.add(db_result)
        await db.commit()
        await db.refresh(db_result)
        
        return db_result
    except Exception as e:
        print(f"Error en submit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[VocationalTestResultResponse])
async def get_history(current_user: User = Depends(deps.get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(VocationalTestResult).where(VocationalTestResult.user_id == current_user.id).order_by(VocationalTestResult.created_at.desc()))
    return result.scalars().all()
