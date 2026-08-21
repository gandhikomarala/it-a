# Comprehensive Unit Test for CoSellPipelineLagExtractor (B2B Cloud Marketplace SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.cloud_marketplace.co_sell_opportunity_pipeline_lag import CoSellPipelineLagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_co_sell_opportunity_pipeline_lag_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CoSellPipelineLagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"co_sell_opportunity_pipeline_lag_signal" in res.columns
    assert f"co_sell_opportunity_pipeline_lag_risk_score" in res.columns
    assert not res[f"co_sell_opportunity_pipeline_lag_signal"].isnull().any()

def test_co_sell_opportunity_pipeline_lag_empty_handling():
    extractor = CoSellPipelineLagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
