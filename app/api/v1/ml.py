from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Body
from app.services.ml_service import ml_service
from app.api import deps
from typing import Any, List, Optional
from pydantic import BaseModel

class TrainRequest(BaseModel):
    filenames: Optional[List[str]] = None

router = APIRouter()

@router.get("/stats")
async def get_stats() -> Any:
    """
    Get current model statistics and training logs.
    """
    stats = ml_service.get_model_stats()
    if not stats:
        raise HTTPException(status_code=404, detail="Model stats not found. Please train the model first.")
    return stats

@router.get("/training-logs")
async def get_training_logs() -> Any:
    """
    Get the raw text logs from the last training execution.
    """
    logs = ml_service.get_training_logs()
    return {"logs": logs}

@router.get("/metrics-explanation")
async def get_metrics_explanation() -> Any:
    """
    Get a human-readable markdown explanation of the machine learning metrics.
    """
    explanation = """
### 🧠 Guía de Métricas del Modelo de IA Vocacional

Para entender el rendimiento de nuestra Inteligencia Artificial, utilizamos métricas estándar de la ciencia de datos. Aquí te explicamos qué significa cada una en el contexto de la orientación vocacional:

#### 1. Accuracy Global (Precisión Exacta)
Es el porcentaje de veces que la IA "acertó" la carrera exacta del estudiante según los datos históricos. 
*Nota: En psicología y ciencias sociales, un Accuracy mayor al 50-60% es considerado excelente, ya que el comportamiento humano es altamente impredecible.*

#### 2. F1-Score Estimado (Equilibrio)
Imagina que la IA es muy buena detectando Ingenieros pero pésima detectando Artistas. El F1-Score castiga esos desequilibrios. Es la media armónica entre Precisión y Cobertura. Si el F1-Score es alto, significa que la IA es "justa" y buena diagnosticando **todas** las categorías por igual.

#### 3. Precisión Global (Calidad de la Predicción)
De todos los estudiantes que la IA diagnosticó como "Ciencias de la Salud", ¿cuántos realmente lo eran? Esta métrica mide cuántos "falsos positivos" arroja el sistema. Una alta precisión significa que cuando la IA sugiere una carrera, puedes confiar en ella ciegamente.

#### 4. Cobertura o Recall (Capacidad de Detección)
De todos los verdaderos estudiantes de "Ciencias de la Salud" que existían en la prueba, ¿cuántos logró encontrar la IA? Mide los "falsos negativos". Una alta cobertura significa que la IA rara vez pasa por alto el talento de un estudiante.

#### 5. AUC-ROC (Área Bajo la Curva)
Es la métrica más avanzada del dashboard. Mide la capacidad del modelo para distinguir entre diferentes carreras sin confundirse. Un valor cercano a 0.5 significa que la IA adivina al azar. Un valor superior a 0.8 significa que la IA tiene una capacidad de discriminación sobresaliente.

#### 6. Soporte (Muestra de Prueba)
Es el número de expedientes de estudiantes reales que se apartaron y se usaron para poner a prueba a la IA después de su entrenamiento.
"""
    return {"markdown": explanation.strip()}


@router.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(...)) -> Any:
    """
    Upload a CSV dataset for training.
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")
    
    content = await file.read()
    path = ml_service.save_dataset(file.filename, content)
    return {"message": f"File {file.filename} uploaded successfully", "path": path}

@router.get("/datasets")
async def list_datasets() -> Any:
    """
    List all uploaded datasets available for training.
    """
    return ml_service.list_datasets()

@router.post("/train")
async def train_model(request: TrainRequest = Body(None)) -> Any:
    """
    Trigger model training using specified datasets (or synthetic data if none provided).
    """
    try:
        filenames = request.filenames if request else None
        stats = await ml_service.train_from_files(filenames=filenames)
        return {"message": "Model trained successfully", "stats": stats}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
