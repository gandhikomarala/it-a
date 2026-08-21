# Unit Test for EngagementMomentumExtractor_Restaurantfranchise (QSR Franchise Store Operations).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.restaurant_franchise.engagement_momentum import EngagementMomentumExtractor_Restaurantfranchise
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_restaurant_franchise_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Restaurantfranchise()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_restaurant_franchise_signal" in res.columns
    assert f"engagement_momentum_restaurant_franchise_risk_score" in res.columns
    assert not res[f"engagement_momentum_restaurant_franchise_signal"].isnull().any()

def test_engagement_momentum_restaurant_franchise_empty():
    extractor = EngagementMomentumExtractor_Restaurantfranchise()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
