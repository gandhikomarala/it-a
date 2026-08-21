# Comprehensive Unit Test for TelematicsSafetyScoreExtractor (Insurance & Actuarial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.insurance_actuarial.telematics_driver_safety_score import TelematicsSafetyScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telematics_driver_safety_score_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelematicsSafetyScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telematics_driver_safety_score_signal" in res.columns
    assert f"telematics_driver_safety_score_risk_score" in res.columns
    assert not res[f"telematics_driver_safety_score_signal"].isnull().any()

def test_telematics_driver_safety_score_empty_handling():
    extractor = TelematicsSafetyScoreExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
