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

# CATEGORIAS ULTRA-SOLIDAS (4 BLOQUES PARA EL 75%)
CAREER_INFO = {
    'Ingeniería y Tecnología': {
        'description': 'Muestras un perfil analítico excepcional con gran capacidad para la resolución de problemas técnicos y tecnológicos.',
        'careers': ['Ingeniería Civil', 'Ingeniería de Sistemas', 'Ingeniería Mecánica', 'Arquitectura', 'Ciencia de Datos']
    },
    'Ciencias de la Salud': {
        'description': 'Tu perfil indica una fuerte vocación hacia el bienestar humano, la investigación biológica y el cuidado médico.',
        'careers': ['Medicina', 'Enfermería', 'Psicología', 'Nutrición', 'Odontología']
    },
    'Artes, Humanidades y Educación': {
        'description': 'Posees una gran sensibilidad artística, habilidades sociales y una fuerte inclinación hacia la expresión creativa y la enseñanza.',
        'careers': ['Diseño Gráfico', 'Artes Escénicas', 'Educación', 'Sociología', 'Comunicación Audiovisual', 'Literatura']
    },
    'Negocios, Gestión y Derecho': {
        'description': 'Tu perfil es estratégico, con grandes habilidades persuasivas y una fuerte capacidad para la gestión organizacional y el marco legal.',
        'careers': ['Administración', 'Derecho', 'Contabilidad', 'Marketing', 'Economía', 'Negocios Internacionales']
    }
}

def format_student_result(r):
    riasec = {"R": 0, "I": 0, "A": 0, "S": 0, "E": 0, "C": 0}
    if r.scores and isinstance(r.scores, dict):
        for k, v in r.scores.items():
            if k and k[0] in riasec and k[1:].isdigit():
                try: riasec[k[0]] += int(v)
                except: pass
    for c in riasec: riasec[c] = int((riasec[c] / 40.0) * 100)
    
    return {
        "id": r.id,
        "scores": r.scores,
        "riasec_percentages": riasec,
        "recommendation": r.recommendation,
        "details": r.details,
        "created_at": r.created_at
    }

@router.post("/submit", response_model=VocationalTestResultResponse)
async def submit_test(
    submission: VocationalTestSubmission,
    current_user: User = Depends(deps.get_current_user),
    db: AsyncSession = Depends(get_db)
):
    try:
        q_result = await db.execute(select(Question))
        q_map = {q.id: q.category for q in q_result.scalars().all()}
        cat_counts = {"Realista": 0, "Investigativo": 0, "Artístico": 0, "Social": 0, "Emprendedor": 0, "Convencional": 0}
        cat_to_letter = {"Realista": "R", "Investigativo": "I", "Artístico": "A", "Social": "S", "Emprendedor": "E", "Convencional": "C"}
        
        raw_scores = {"age": submission.age, "gender": submission.gender, "education": submission.education}
        for ans in submission.answers:
            cat_name = q_map.get(ans.question_id)
            if cat_name in cat_counts:
                cat_counts[cat_name] += 1
                letter = cat_to_letter[cat_name]
                raw_scores[f"{letter}{cat_counts[cat_name]}"] = ans.value
        
        explanation = ml_service.explain_prediction(raw_scores)
        if not explanation: raise HTTPException(status_code=500, detail="Modelo no entrenado")
        recommendation = str(explanation["insights"]["prediction"])
        
        info = CAREER_INFO.get(recommendation, {'description': 'Perfil vocacional identificado.', 'careers': []})
        
        # Extraer Análisis Avanzado de la IA (Opción secundaria y multipotencialidad)
        insights = explanation.get("insights", {})
        second_opt = insights.get("second_option")
        is_multi = insights.get("is_multipotential")
        
        analysis_text = ""
        if second_opt:
            if is_multi:
                analysis_text = f" Análisis Extra de la IA: Tu perfil es altamente Multipotencial. Aunque tu principal ruta es {recommendation}, también tienes aptitudes sobresalientes para estudiar {second_opt['career']} (con un {second_opt['confidence']}% de afinidad secundaria)."
            else:
                analysis_text = f" Análisis Extra de la IA: Como ruta alternativa, tu perfil también muestra compatibilidad con {second_opt['career']}."

        details = f"{info['description']} Carreras sugeridas principales: {', '.join(info['careers'])}.{analysis_text}"
        
        db_result = VocationalTestResult(user_id=current_user.id, scores=raw_scores, recommendation=recommendation, details=details)
        db.add(db_result)
        await db.commit()
        await db.refresh(db_result)
        return format_student_result(db_result)

    except Exception as e:
        print(f"Error en submit: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history", response_model=List[VocationalTestResultResponse])
async def get_history(current_user: User = Depends(deps.get_current_user), db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(VocationalTestResult).where(VocationalTestResult.user_id == current_user.id).order_by(VocationalTestResult.created_at.desc()))
    return [format_student_result(r) for r in result.scalars().all()]

