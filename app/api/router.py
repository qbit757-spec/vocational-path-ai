from fastapi import APIRouter
from app.api.v1 import auth, test, ml

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["auth"])
api_router.include_router(test.router, prefix="/test", tags=["vocational-test"])
api_router.include_router(ml.router, prefix="/ml", tags=["machine-learning"])
