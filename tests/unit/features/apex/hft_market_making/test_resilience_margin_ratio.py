# Unit Test for ResilienceMarginRatioExtractor_Hftmarketmaking (High-Frequency Trading & Market Making).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hft_market_making.resilience_margin_ratio import ResilienceMarginRatioExtractor_Hftmarketmaking
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_resilience_margin_ratio_hft_market_making_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ResilienceMarginRatioExtractor_Hftmarketmaking()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"resilience_margin_ratio_hft_market_making_signal" in res.columns
    assert f"resilience_margin_ratio_hft_market_making_risk_score" in res.columns
    assert not res[f"resilience_margin_ratio_hft_market_making_signal"].isnull().any()

def test_resilience_margin_ratio_hft_market_making_empty():
    extractor = ResilienceMarginRatioExtractor_Hftmarketmaking()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
