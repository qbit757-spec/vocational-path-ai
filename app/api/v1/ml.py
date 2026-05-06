from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from app.services.ml_service import ml_service
from app.api import deps
from typing import Any, List

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
async def train_model(filenames: List[str] = None) -> Any:
    """
    Trigger model training using specified datasets (or synthetic data if none provided).
    """
    try:
        stats = await ml_service.train_from_files(filenames=filenames)
        return {"message": "Model trained successfully", "stats": stats}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")
