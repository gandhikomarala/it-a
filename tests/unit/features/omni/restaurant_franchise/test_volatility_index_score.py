# Unit Test for VolatilityIndexScoreExtractor_Restaurantfranchise (QSR Franchise Store Operations).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.restaurant_franchise.volatility_index_score import VolatilityIndexScoreExtractor_Restaurantfranchise
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_restaurant_franchise_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Restaurantfranchise()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_restaurant_franchise_signal" in res.columns
    assert f"volatility_index_score_restaurant_franchise_risk_score" in res.columns
    assert not res[f"volatility_index_score_restaurant_franchise_signal"].isnull().any()

def test_volatility_index_score_restaurant_franchise_empty():
    extractor = VolatilityIndexScoreExtractor_Restaurantfranchise()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
