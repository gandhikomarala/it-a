# Unit Test for EngagementMomentumExtractor_Pensionactuarial (Defined Benefit Pension Fund Actuarial).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.pension_actuarial.engagement_momentum import EngagementMomentumExtractor_Pensionactuarial
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_pension_actuarial_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Pensionactuarial()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_pension_actuarial_signal" in res.columns
    assert f"engagement_momentum_pension_actuarial_risk_score" in res.columns
    assert not res[f"engagement_momentum_pension_actuarial_signal"].isnull().any()

def test_engagement_momentum_pension_actuarial_empty():
    extractor = EngagementMomentumExtractor_Pensionactuarial()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
