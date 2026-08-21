# Multi-metric model evaluation engine.
import time
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss, confusion_matrix
)
from packages.schemas.experiment import ExperimentMetricSchema
from packages.logging.logger import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    @staticmethod
    def evaluate(
        model: Any,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        training_time_seconds: float = 0.0
    ) -> ExperimentMetricSchema:
        t0 = time.perf_counter()
        y_prob = model.predict_proba(X_test)
        inference_duration = time.perf_counter() - t0
        latency_ms = (inference_duration / len(X_test)) * 1000.0 if len(X_test) > 0 else 0.0

        y_pred = (y_prob >= 0.50).astype(int)

        acc = float(accuracy_score(y_test, y_pred))
        prec = float(precision_score(y_test, y_pred, zero_division=0))
        rec = float(recall_score(y_test, y_pred, zero_division=0))
        f1 = float(f1_score(y_test, y_pred, zero_division=0))
        
        try:
            roc_auc = float(roc_auc_score(y_test, y_prob))
        except Exception:
            roc_auc = 0.50

        try:
            pr_auc = float(average_precision_score(y_test, y_prob))
        except Exception:
            pr_auc = 0.0

        brier = float(brier_score_loss(y_test, y_prob))

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

        cm_dict = {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        }

        return ExperimentMetricSchema(
            accuracy=round(acc, 4),
            precision=round(prec, 4),
            recall=round(rec, 4),
            f1_score=round(f1, 4),
            roc_auc=round(roc_auc, 4),
            pr_auc=round(pr_auc, 4),
            brier_score=round(brier, 4),
            confusion_matrix=cm_dict,
            training_time_seconds=round(training_time_seconds, 2),
            inference_latency_ms=round(latency_ms, 3)
        )
