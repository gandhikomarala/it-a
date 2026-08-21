# FastAPI Application Factory and Root Orchestrator.
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from packages.configuration.settings import get_settings
from packages.logging.logger import setup_logging, get_logger
from backend.database.session import init_db
from backend.core.middleware import RequestIDMiddleware

from apps.api.v1.auth import router as auth_router
from apps.api.v1.customers import router as customers_router
from apps.api.v1.datasets import router as datasets_router
from apps.api.v1.models import router as models_router
from apps.api.v1.predictions import router as predictions_router
from apps.api.v1.analytics import router as analytics_router
from apps.api.v1.system import router as system_router

settings = get_settings()
setup_logging(settings.LOG_LEVEL)
logger = get_logger("app.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing platform backend and database...")
    try:
        await init_db()
    except Exception as e:
        logger.warning(f"Database init skipped or handled via migrations: {e}")
    yield
    logger.info("Platform shutting down gracefully.")

app = FastAPI(
    title=settings.APP_NAME,
    version="1.0.0",
    description="Enterprise Customer Churn Prediction & MLOps Platform REST API",
    lifespan=lifespan
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router, prefix=settings.API_V1_STR)
app.include_router(customers_router, prefix=settings.API_V1_STR)
app.include_router(datasets_router, prefix=settings.API_V1_STR)
app.include_router(models_router, prefix=settings.API_V1_STR)
app.include_router(predictions_router, prefix=settings.API_V1_STR)
app.include_router(analytics_router, prefix=settings.API_V1_STR)
app.include_router(system_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "platform": settings.APP_NAME,
        "version": "1.0.0",
        "status": "ONLINE",
        "docs_url": "/docs",
        "api_v1": settings.API_V1_STR
    }
