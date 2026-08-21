# Automated Retraining Decision and Safety Policy Engine.
from typing import Dict, Any, Optional
import pandas as pd
from packages.logging.logger import get_logger
from ml.pipelines.retraining_pipeline import AutomatedRetrainingPipeline
from ml.monitoring.drift_detector import StatisticalDriftDetector

logger = get_logger("service.retraining")

class RetrainingService:
    @staticmethod
    async def evaluate_and_trigger(
        current_data: pd.DataFrame,
        reference_data: pd.DataFrame,
        active_model_auc: float = 0.88
    ) -> Dict[str, Any]:
        logger.info("Evaluating automated retraining policy...")
        
        # 1. Detect drift
        detector = StatisticalDriftDetector()
        drift_report = detector.calculate_drift(reference_data, current_data)

        # 2. Check if retraining threshold triggered
        should_retrain = (
            drift_report.max_psi >= 0.25 or
            drift_report.overall_drift_status.value == "CRITICAL"
        )

        result = {
            "drift_status": drift_report.overall_drift_status.value,
            "max_psi": drift_report.max_psi,
            "retraining_triggered": should_retrain,
            "promotion_status": "SKIPPED"
        }

        if should_retrain:
            logger.info("Automated retraining triggered by drift policy. Launching pipeline...")
            retrain_res = AutomatedRetrainingPipeline.run_pipeline(
                current_data=current_data,
                reference_data=reference_data,
                active_model_roc_auc=active_model_auc
            )
            result["promotion_status"] = "PROMOTED" if retrain_res["promoted_to_production"] else "HELD_IN_STAGING"
            result["candidate_metrics"] = retrain_res["candidate_metrics"].model_dump()

        return result
