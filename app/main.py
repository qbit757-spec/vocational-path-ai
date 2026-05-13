from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.api.router import api_router
from app.core.config import settings
from app.ml.predictor import predictor
import uvicorn

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_origin(request, call_next):
    origin = request.headers.get("origin")
    print(f"Incoming request from origin: {origin}")
    response = await call_next(request)
    return response

app.include_router(api_router, prefix=settings.API_V1_STR)

@app.on_event("startup")
async def startup_event():
    # Load ML model on startup
    print(f"CORS allowed origins: {settings.BACKEND_CORS_ORIGINS}")
    print("Loading ML model...")
    if not predictor.load_model():
        print("Model not found. Please run scripts/train_model.py first.")

@app.get("/")
async def root():
    return {"message": "API del Sistema web para la elección de carreras profesionales - UGEL 03 Lima Centro"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
