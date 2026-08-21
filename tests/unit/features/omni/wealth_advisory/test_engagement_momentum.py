# Unit Test for EngagementMomentumExtractor_Wealthadvisory (Private Wealth Advisory & Estate Planning).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.wealth_advisory.engagement_momentum import EngagementMomentumExtractor_Wealthadvisory
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_wealth_advisory_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Wealthadvisory()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_wealth_advisory_signal" in res.columns
    assert f"engagement_momentum_wealth_advisory_risk_score" in res.columns
    assert not res[f"engagement_momentum_wealth_advisory_signal"].isnull().any()

def test_engagement_momentum_wealth_advisory_empty():
    extractor = EngagementMomentumExtractor_Wealthadvisory()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
