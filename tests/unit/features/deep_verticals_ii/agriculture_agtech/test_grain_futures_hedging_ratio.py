# Comprehensive Unit Test for GrainFuturesHedgingRatioExtractor (Agriculture & Precision Farming).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.agriculture_agtech.grain_futures_hedging_ratio import GrainFuturesHedgingRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_grain_futures_hedging_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = GrainFuturesHedgingRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"grain_futures_hedging_ratio_signal" in res.columns
    assert f"grain_futures_hedging_ratio_risk_score" in res.columns
    assert not res[f"grain_futures_hedging_ratio_signal"].isnull().any()

def test_grain_futures_hedging_ratio_empty_handling():
    extractor = GrainFuturesHedgingRatioExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
