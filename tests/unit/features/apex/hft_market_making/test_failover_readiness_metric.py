# Unit Test for FailoverReadinessMetricExtractor_Hftmarketmaking (High-Frequency Trading & Market Making).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hft_market_making.failover_readiness_metric import FailoverReadinessMetricExtractor_Hftmarketmaking
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_hft_market_making_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Hftmarketmaking()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_hft_market_making_signal" in res.columns
    assert f"failover_readiness_metric_hft_market_making_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_hft_market_making_signal"].isnull().any()

def test_failover_readiness_metric_hft_market_making_empty():
    extractor = FailoverReadinessMetricExtractor_Hftmarketmaking()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
