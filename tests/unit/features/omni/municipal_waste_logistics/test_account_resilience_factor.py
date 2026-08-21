# Unit Test for AccountResilienceFactorExtractor_Municipalwastelogistics (Municipal Smart Waste Routing).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.municipal_waste_logistics.account_resilience_factor import AccountResilienceFactorExtractor_Municipalwastelogistics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_municipal_waste_logistics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Municipalwastelogistics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_municipal_waste_logistics_signal" in res.columns
    assert f"account_resilience_factor_municipal_waste_logistics_risk_score" in res.columns
    assert not res[f"account_resilience_factor_municipal_waste_logistics_signal"].isnull().any()

def test_account_resilience_factor_municipal_waste_logistics_empty():
    extractor = AccountResilienceFactorExtractor_Municipalwastelogistics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
