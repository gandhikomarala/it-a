# Unit Test for AccountResilienceFactorExtractor_Restaurantfranchise (QSR Franchise Store Operations).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.restaurant_franchise.account_resilience_factor import AccountResilienceFactorExtractor_Restaurantfranchise
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_restaurant_franchise_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Restaurantfranchise()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_restaurant_franchise_signal" in res.columns
    assert f"account_resilience_factor_restaurant_franchise_risk_score" in res.columns
    assert not res[f"account_resilience_factor_restaurant_franchise_signal"].isnull().any()

def test_account_resilience_factor_restaurant_franchise_empty():
    extractor = AccountResilienceFactorExtractor_Restaurantfranchise()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
