# Unit Test for LifecycleBurnRateExtractor_Hftmarketmaking (High-Frequency Trading & Market Making).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hft_market_making.lifecycle_burn_rate import LifecycleBurnRateExtractor_Hftmarketmaking
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_burn_rate_hft_market_making_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LifecycleBurnRateExtractor_Hftmarketmaking()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lifecycle_burn_rate_hft_market_making_signal" in res.columns
    assert f"lifecycle_burn_rate_hft_market_making_risk_score" in res.columns
    assert not res[f"lifecycle_burn_rate_hft_market_making_signal"].isnull().any()

def test_lifecycle_burn_rate_hft_market_making_empty():
    extractor = LifecycleBurnRateExtractor_Hftmarketmaking()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
