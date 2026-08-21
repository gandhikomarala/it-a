# Behavioral feature extractor calculating high-order churn indicators.
import pandas as pd
import numpy as np

class BehavioralFeatureExtractor:
    @staticmethod
    def extract_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        if "tenure_months" in df.columns:
            df["is_new_customer"] = (df["tenure_months"] <= 3).astype(int)
            df["is_veteran_customer"] = (df["tenure_months"] >= 24).astype(int)

        if "complaint_count" in df.columns and "satisfaction_score" in df.columns:
            df["support_dissatisfaction_index"] = (df["complaint_count"] * 1.5) + (5.0 - df["satisfaction_score"])

        if "days_since_last_login" in df.columns and "daily_usage_hours" in df.columns:
            df["engagement_decay_score"] = (df["days_since_last_login"] / 7.0) / (df["daily_usage_hours"] + 0.1)

        if all(c in df.columns for c in ["contract_type", "payment_failures_count", "days_since_last_login"]):
            is_m2m = (df["contract_type"] == "Month-to-Month").astype(float)
            fails = df["payment_failures_count"].astype(float)
            inactivity = (df["days_since_last_login"] > 7).astype(float)
            df["vulnerability_score"] = (is_m2m * 2.0) + (fails * 1.5) + (inactivity * 1.2)

        return df
