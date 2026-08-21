"""Pydantic BaseSettings providing strongly typed configuration across the platform."""
import os
from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # Application & Environment
    APP_NAME: str = "Enterprise Customer Churn Prediction & MLOps Platform"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    API_V1_STR: str = "/api/v1"

    # Security & Cryptography
    SECRET_KEY: str = "super-secret-system-key-change-this-in-production-min-32-chars-long!"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    PASSWORD_RESET_TOKEN_EXPIRE_HOURS: int = 24
    ARGON2_TIME_COST: int = 2
    ARGON2_MEMORY_COST: int = 65536
    ARGON2_PARALLELISM: int = 1

    # Database Configuration
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "churn_platform"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/churn_platform"
    DATABASE_URL_SYNC: str = "postgresql://postgres:postgres@localhost:5432/churn_platform"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10
    DB_POOL_TIMEOUT: int = 30

    # Redis & Caching
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_DEFAULT_TIMEOUT: int = 300

    # Celery Background Tasks
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_TASK_TRACK_STARTED: bool = True
    CELERY_TASK_TIME_LIMIT: int = 3600  # 1 hour max
    CELERY_MAX_RETRIES: int = 3

    # Storage & Artifacts
    ARTIFACT_STORAGE_PATH: str = "./artifacts/models"
    DATASET_STORAGE_PATH: str = "./artifacts/datasets"
    REPORT_STORAGE_PATH: str = "./artifacts/reports"
    MAX_UPLOAD_SIZE_BYTES: int = 250 * 1024 * 1024  # 250 MB
    ALLOWED_UPLOAD_EXTENSIONS: List[str] = [".csv", ".parquet", ".json"]

    # ML & Risk Thresholds
    DEFAULT_RISK_LOW_THRESHOLD: float = 0.30
    DEFAULT_RISK_HIGH_THRESHOLD: float = 0.70
    RETRAINING_DRIFT_THRESHOLD: float = 0.25
    RETRAINING_PERFORMANCE_DROP_THRESHOLD: float = 0.05
    RETRAINING_MIN_SAMPLES: int = 1000

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE_ANONYMOUS: int = 30
    RATE_LIMIT_PER_MINUTE_AUTHENTICATED: int = 300
    RATE_LIMIT_PER_MINUTE_TRAINING: int = 10

    # CORS
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000"
    ]

    # AWS Deployment Settings
    AWS_REGION: str = "us-east-1"
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    S3_BUCKET_NAME: Optional[str] = "churn-mlops-artifacts"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

@lru_cache()
def get_settings() -> Settings:
    return Settings()
