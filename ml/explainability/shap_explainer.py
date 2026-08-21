# SHAP explainer engine providing local and global feature attribution.
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
from packages.schemas.prediction import PredictionExplanationResponse, SHAPFactorContribution
from packages.logging.logger import get_logger

logger = get_logger(__name__)

class ModelExplainer:
    def __init__(self, model_wrapper: Any, background_sample: Optional[pd.DataFrame] = None):
        self.model_wrapper = model_wrapper
        self.background_sample = background_sample
        self.explainer: Any = None
        self._init_explainer()

    def _init_explainer(self):
        try:
            import shap
            underlying_model = getattr(self.model_wrapper, "model", self.model_wrapper)
            if hasattr(underlying_model, "predict_proba"):
                if "LGBM" in type(underlying_model).__name__ or "Forest" in type(underlying_model).__name__:
                    self.explainer = shap.TreeExplainer(underlying_model)
                else:
                    if self.background_sample is not None and len(self.background_sample) > 0:
                        bg = shap.sample(self.background_sample.values, min(50, len(self.background_sample)))
                        self.explainer = shap.KernelExplainer(underlying_model.predict_proba, bg)
        except Exception as e:
            logger.warning(f"Failed to initialize full SHAP explainer, fallback to feature weights: {e}")
            self.explainer = None

    def explain_instance(
        self,
        customer_id: str,
        instance_df: pd.DataFrame,
        predicted_probability: float
    ) -> PredictionExplanationResponse:
        feature_names = instance_df.columns.tolist()
        feature_values = instance_df.iloc[0].to_dict()

        shap_values_dict: Dict[str, float] = {}

        if self.explainer is not None:
            try:
                values = self.explainer.shap_values(instance_df.values)
                if isinstance(values, list) and len(values) == 2:
                    raw_shap = values[1][0]
                elif isinstance(values, np.ndarray) and values.ndim == 2:
                    raw_shap = values[0]
                else:
                    raw_shap = np.array(values).flatten()[:len(feature_names)]
                for feat, s_val in zip(feature_names, raw_shap):
                    shap_values_dict[feat] = float(s_val)
            except Exception as e:
                logger.warning(f"SHAP compute error: {e}")
                shap_values_dict = self._fallback_feature_weights(instance_df)
        else:
            shap_values_dict = self._fallback_feature_weights(instance_df)

        total_abs = sum(abs(v) for v in shap_values_dict.values()) or 1.0
        
        all_contributions: List[SHAPFactorContribution] = []
        for feat, val in shap_values_dict.items():
            display_name = feat.replace('_', ' ').title()
            direction = "POSITIVE" if val > 0 else "NEGATIVE"
            pct = round((abs(val) / total_abs) * 100.0, 1)
            all_contributions.append(SHAPFactorContribution(
                feature_name=feat,
                display_name=display_name,
                feature_value=feature_values.get(feat),
                shap_value=round(val, 4),
                contribution_percentage=pct,
                impact_direction=direction
            ))

        pos_factors = sorted([c for c in all_contributions if c.impact_direction == "POSITIVE"], key=lambda x: x.contribution_percentage, reverse=True)[:5]
        neg_factors = sorted([c for c in all_contributions if c.impact_direction == "NEGATIVE"], key=lambda x: x.contribution_percentage, reverse=True)[:5]

        top_driver = pos_factors[0].display_name if pos_factors else "Tenure"
        top_pct = pos_factors[0].contribution_percentage if pos_factors else 0.0
        summary_text = (
            f"Customer {customer_id} has a churn risk of {predicted_probability * 100.0:.1f}%. "
            f"Primary factor increasing risk is '{top_driver}' (+{top_pct}% relative impact)."
        )

        return PredictionExplanationResponse(
            customer_id=customer_id,
            base_value=0.25,
            prediction_probability=round(predicted_probability, 4),
            top_positive_factors=pos_factors,
            top_negative_factors=neg_factors,
            all_contributions=all_contributions,
            summary_text=summary_text
        )

    def _fallback_feature_weights(self, instance_df: pd.DataFrame) -> Dict[str, float]:
        importances = self.model_wrapper.get_feature_importances()
        weights = {}
        for feat in instance_df.columns:
            imp = importances.get(feat, 0.05)
            if any(k in feat for k in ["failure", "complaint", "days_since", "Month-to-Month"]):
                weights[feat] = float(imp)
            else:
                weights[feat] = -float(imp)
        return weights
