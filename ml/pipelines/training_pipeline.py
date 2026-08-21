# Complete training, evaluation, comparison, and registration workflow pipeline.
from typing import Dict, Any, List
import pandas as pd
from ml.training.orchestrator import TrainingOrchestrator
from ml.registry.registry_manager import ModelRegistryManager
from packages.logging.logger import get_logger

logger = get_logger(__name__)

class FullTrainingPipeline:
    @staticmethod
    def run_experiment(
        df: pd.DataFrame,
        experiment_name: str,
        algorithms: List[str] = None,
        training_mode: str = "STANDARD"
    ) -> Dict[str, Any]:
        if algorithms is None:
            algorithms = ["LogisticRegression", "RandomForest", "LightGBM"]

        results: Dict[str, Any] = {}
        best_auc = -1.0
        best_algo = None
        best_pipeline = None
        best_model = None
        best_metrics = None

        for algo in algorithms:
            pipeline, model, metrics = TrainingOrchestrator.train_and_evaluate(
                df=df,
                algorithm=algo,
                training_mode=training_mode
            )
            results[algo] = {
                "metrics": metrics,
                "pipeline": pipeline,
                "model": model
            }
            if metrics.roc_auc > best_auc:
                best_auc = metrics.roc_auc
                best_algo = algo
                best_pipeline = pipeline
                best_model = model
                best_metrics = metrics

        registry = ModelRegistryManager()
        saved_info = registry.save_model_artifact(
            model_name=best_algo,
            version=1,
            pipeline=best_pipeline,
            model_wrapper=best_model,
            metadata={"experiment_name": experiment_name, "best_roc_auc": best_auc}
        )

        return {
            "experiment_name": experiment_name,
            "best_algorithm": best_algo,
            "best_metrics": best_metrics,
            "artifact_info": saved_info,
            "all_results": {k: v["metrics"] for k, v in results.items()}
        }
