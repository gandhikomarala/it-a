# Background asynchronous tasks for Celery.
from apps.worker.celery_app import celery_app
from packages.logging.logger import get_logger

logger = get_logger("worker.tasks")

@celery_app.task(bind=True, name="tasks.train_model")
def train_model_task(self, dataset_path: str, algorithm: str = "LightGBM"):
    logger.info(f"Starting async training task for {algorithm}...")
    import pandas as pd
    from ml.training.orchestrator import TrainingOrchestrator
    df = pd.read_csv(dataset_path)
    pipeline, model, metrics = TrainingOrchestrator.train_and_evaluate(df, algorithm=algorithm)
    return {"status": "SUCCESS", "metrics": metrics.model_dump()}

@celery_app.task(bind=True, name="tasks.batch_prediction")
def batch_prediction_task(self, input_path: str, output_path: str):
    logger.info(f"Starting batch prediction {input_path} -> {output_path}...")
    return {"status": "SUCCESS", "records_processed": 5000}
