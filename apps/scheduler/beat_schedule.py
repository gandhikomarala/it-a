# Periodic Celery Beat schedules.
from celery.schedules import crontab
from apps.worker.celery_app import celery_app

celery_app.conf.beat_schedule = {
    "hourly-drift-check": {
        "task": "tasks.run_drift_check",
        "schedule": crontab(minute=0),
    },
    "weekly-retraining-evaluation": {
        "task": "tasks.evaluate_retraining_trigger",
        "schedule": crontab(hour=0, minute=0, day_of_week=0),
    }
}
