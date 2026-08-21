# Comprehensive ML training orchestrator linking preprocessing, tuning, and evaluation.
import time
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ml.preprocessing.pipeline import PreprocessingPipeline
from ml.models.logistic_regression import LogisticRegressionModel
from ml.models.random_forest import RandomForestModel
from ml.models.gradient_boosting import GradientBoostingModel
from ml.models.lightgbm_model import LightGBMModel
from ml.models.ensemble import EnsembleModel
from ml.evaluation.evaluator import ModelEvaluator
from ml.training.hyperopt import HyperparameterTuner
from packages.schemas.experiment import ExperimentMetricSchema
from packages.logging.logger import get_logger, LogContext

logger = get_logger(__name__)

ALGORITHM_MAP = {
    "LogisticRegression": LogisticRegressionModel,
    "RandomForest": RandomForestModel,
    "GradientBoosting": GradientBoostingModel,
    "LightGBM": LightGBMModel,
    "Ensemble": EnsembleModel
}

class TrainingOrchestrator:
    @staticmethod
    def train_and_evaluate(
        df: pd.DataFrame,
        algorithm: str = "LightGBM",
        hyperparameters: Dict[str, Any] = None,
        training_mode: str = "STANDARD",
        test_size: float = 0.20,
        random_state: int = 42
    ) -> Tuple[PreprocessingPipeline, Any, ExperimentMetricSchema]:
        if "churn" not in df.columns:
            raise ValueError("Target column 'churn' not found in training dataset.")

        with LogContext(logger, f"Train {algorithm} in {training_mode} mode"):
            y = df["churn"].values.astype(int)
            X = df.drop(columns=["churn"])

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )

            pipeline = PreprocessingPipeline()
            X_train_trans = pipeline.fit_transform(X_train)
            X_test_trans = pipeline.transform(X_test)

            default_params = HyperparameterTuner.get_search_grid(algorithm, training_mode)
            if hyperparameters:
                default_params.update(hyperparameters)

            model_cls = ALGORITHM_MAP.get(algorithm, LightGBMModel)
            model_wrapper = model_cls(hyperparameters=default_params)

            t0 = time.perf_counter()
            model_wrapper.fit(X_train_trans, y_train)
            training_duration = time.perf_counter() - t0

            metrics = ModelEvaluator.evaluate(
                model=model_wrapper,
                X_test=X_test_trans,
                y_test=y_test,
                training_time_seconds=training_duration
            )

            return pipeline, model_wrapper, metrics
