# Comprehensive Unit Test for VibrationAnomalyScoreExtractor (Manufacturing & Industrial IoT).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.manufacturing_iiot.vibration_acoustic_anomaly_score import VibrationAnomalyScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_vibration_acoustic_anomaly_score_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VibrationAnomalyScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"vibration_acoustic_anomaly_score_signal" in res.columns
    assert f"vibration_acoustic_anomaly_score_risk_score" in res.columns
    assert not res[f"vibration_acoustic_anomaly_score_signal"].isnull().any()

def test_vibration_acoustic_anomaly_score_empty_handling():
    extractor = VibrationAnomalyScoreExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
