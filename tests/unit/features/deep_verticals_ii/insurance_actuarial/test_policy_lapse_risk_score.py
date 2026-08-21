# Comprehensive Unit Test for PolicyLapseRiskScoreExtractor (Insurance & Actuarial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.insurance_actuarial.policy_lapse_risk_score import PolicyLapseRiskScoreExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_policy_lapse_risk_score_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PolicyLapseRiskScoreExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"policy_lapse_risk_score_signal" in res.columns
    assert f"policy_lapse_risk_score_risk_score" in res.columns
    assert not res[f"policy_lapse_risk_score_signal"].isnull().any()

def test_policy_lapse_risk_score_empty_handling():
    extractor = PolicyLapseRiskScoreExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
