# Automated retraining pipeline with production comparison safeguards.
from typing import Dict, Any
import pandas as pd
from packages.shared.constants import RETRAINING_POLICIES
from ml.training.orchestrator import TrainingOrchestrator
from ml.registry.registry_manager import ModelRegistryManager
from packages.logging.logger import get_logger

logger = get_logger(__name__)

class AutomatedRetrainingPipeline:
    @staticmethod
    def execute_retraining(
        df_new: pd.DataFrame,
        current_production_auc: float = 0.82
    ) -> Dict[str, Any]:
        logger.info("Executing automated retraining pipeline...")

        pipeline, model, metrics = TrainingOrchestrator.train_and_evaluate(
            df=df_new,
            algorithm="LightGBM",
            training_mode="FULL"
        )

        candidate_auc = metrics.roc_auc
        delta = candidate_auc - current_production_auc

        min_auc_passed = candidate_auc >= RETRAINING_POLICIES["MIN_TEST_ROC_AUC"]
        not_degraded = delta >= -RETRAINING_POLICIES["MAX_PERFORMANCE_DROP_PCT"]
        is_superior = delta >= RETRAINING_POLICIES["REQUIRED_SUPERIORITY_PCT"]

        all_passed = min_auc_passed and not_degraded
        recommendation = "PROMOTE" if is_superior and all_passed else ("MANUAL_REVIEW" if all_passed else "REJECT")

        registry = ModelRegistryManager()
        artifact_info = registry.save_model_artifact(
            model_name="LightGBM_Retrained",
            version=2,
            pipeline=pipeline,
            model_wrapper=model,
            metadata={"retrained": True, "roc_auc": candidate_auc}
        )

        return {
            "candidate_roc_auc": candidate_auc,
            "production_roc_auc": current_production_auc,
            "delta_roc_auc": round(delta, 4),
            "safeguards_passed": all_passed,
            "recommended_action": recommendation,
            "artifact_info": artifact_info,
            "metrics": metrics
        }
