# Unit Test for AccountResilienceFactorExtractor_Cdnedgestreaming (Global CDN Video Edge Caching).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cdn_edge_streaming.account_resilience_factor import AccountResilienceFactorExtractor_Cdnedgestreaming
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_cdn_edge_streaming_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Cdnedgestreaming()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_cdn_edge_streaming_signal" in res.columns
    assert f"account_resilience_factor_cdn_edge_streaming_risk_score" in res.columns
    assert not res[f"account_resilience_factor_cdn_edge_streaming_signal"].isnull().any()

def test_account_resilience_factor_cdn_edge_streaming_empty():
    extractor = AccountResilienceFactorExtractor_Cdnedgestreaming()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
