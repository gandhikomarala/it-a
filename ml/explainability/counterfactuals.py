# Actionable counterfactual recommendations engine for retention.
from typing import Dict, Any, List
import numpy as np
import pandas as pd

class CounterfactualExplainer:
    # Generates minimum actionable changes required to flip a customer from High-Risk to Low-Risk.
    def __init__(self, model_wrapper, feature_pipeline):
        self.model_wrapper = model_wrapper
        self.pipeline = feature_pipeline

    def generate_recommendations(
        self,
        customer_row: pd.Series,
        target_risk_probability: float = 0.25
    ) -> List[Dict[str, Any]]:
        actions = []
        
        # 1. Contract modification scenario
        if customer_row.get("contract_type") == "Month-to-Month":
            c_test = customer_row.to_frame().T.copy()
            c_test["contract_type"] = "One-Year"
            X_t = self.pipeline.transform(c_test)
            p_new = float(self.model_wrapper.predict_proba(X_t)[:, 1][0])
            actions.append({
                "action": "Upgrade contract to One-Year Plan with 15% promotional discount",
                "simulated_churn_probability": round(p_new, 4),
                "risk_reduction_pct": round((customer_row.get("latest_churn_probability", 0.80) - p_new) * 100, 1)
            })

        # 2. Dedicated CSAT Outreach scenario
        if customer_row.get("satisfaction_score", 5.0) < 3.5:
            c_test = customer_row.to_frame().T.copy()
            c_test["satisfaction_score"] = 4.5
            c_test["complaint_count"] = 0
            X_t = self.pipeline.transform(c_test)
            p_new = float(self.model_wrapper.predict_proba(X_t)[:, 1][0])
            actions.append({
                "action": "Assign Senior Account Manager to resolve open complaint and restore CSAT to 4.5",
                "simulated_churn_probability": round(p_new, 4),
                "risk_reduction_pct": round((customer_row.get("latest_churn_probability", 0.80) - p_new) * 100, 1)
            })

        return actions
