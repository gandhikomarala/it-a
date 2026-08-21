# Comprehensive Unit Test for MeteredDimensionUsageBurstsExtractor (B2B Cloud Marketplace SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.cloud_marketplace.metered_dimension_usage_bursts import MeteredDimensionUsageBurstsExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_metered_dimension_usage_bursts_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MeteredDimensionUsageBurstsExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"metered_dimension_usage_bursts_signal" in res.columns
    assert f"metered_dimension_usage_bursts_risk_score" in res.columns
    assert not res[f"metered_dimension_usage_bursts_signal"].isnull().any()

def test_metered_dimension_usage_bursts_empty_handling():
    extractor = MeteredDimensionUsageBurstsExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
