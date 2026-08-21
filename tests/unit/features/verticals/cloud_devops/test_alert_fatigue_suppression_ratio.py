# Unit Test for AlertFatigueSuppressionRatioExtractor (Cloud Infrastructure & DevOps).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cloud_devops.alert_fatigue_suppression_ratio import AlertFatigueSuppressionRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_alert_fatigue_suppression_ratio_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AlertFatigueSuppressionRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"alert_fatigue_suppression_ratio_signal" in res.columns
    assert f"alert_fatigue_suppression_ratio_risk_score" in res.columns
    assert not res[f"alert_fatigue_suppression_ratio_signal"].isnull().any()

def test_alert_fatigue_suppression_ratio_empty_dataframe():
    extractor = AlertFatigueSuppressionRatioExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
