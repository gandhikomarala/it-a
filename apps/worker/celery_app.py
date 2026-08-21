# Celery application instance configuration.
from celery import Celery
from packages.configuration.settings import get_settings

settings = get_settings()

celery_app = Celery(
    "churn_mlops_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    worker_prefetch_multiplier=1
)
