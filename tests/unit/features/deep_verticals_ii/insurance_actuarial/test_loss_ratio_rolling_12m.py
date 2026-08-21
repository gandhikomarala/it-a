# Comprehensive Unit Test for LossRatioRolling12mExtractor (Insurance & Actuarial SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.insurance_actuarial.loss_ratio_rolling_12m import LossRatioRolling12mExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_loss_ratio_rolling_12m_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LossRatioRolling12mExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"loss_ratio_rolling_12m_signal" in res.columns
    assert f"loss_ratio_rolling_12m_risk_score" in res.columns
    assert not res[f"loss_ratio_rolling_12m_signal"].isnull().any()

def test_loss_ratio_rolling_12m_empty_handling():
    extractor = LossRatioRolling12mExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
