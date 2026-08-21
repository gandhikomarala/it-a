# Unit Test for EngagementMomentumExtractor_Luxuryfashiondirect (Luxury Fashion Direct-to-Consumer).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.luxury_fashion_direct.engagement_momentum import EngagementMomentumExtractor_Luxuryfashiondirect
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_luxury_fashion_direct_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Luxuryfashiondirect()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_luxury_fashion_direct_signal" in res.columns
    assert f"engagement_momentum_luxury_fashion_direct_risk_score" in res.columns
    assert not res[f"engagement_momentum_luxury_fashion_direct_signal"].isnull().any()

def test_engagement_momentum_luxury_fashion_direct_empty():
    extractor = EngagementMomentumExtractor_Luxuryfashiondirect()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
