# Comprehensive Unit Test for SeedVarietyYieldDeltaExtractor (Agriculture & Precision Farming).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.agriculture_agtech.seed_variety_yield_delta import SeedVarietyYieldDeltaExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_seed_variety_yield_delta_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SeedVarietyYieldDeltaExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"seed_variety_yield_delta_signal" in res.columns
    assert f"seed_variety_yield_delta_risk_score" in res.columns
    assert not res[f"seed_variety_yield_delta_signal"].isnull().any()

def test_seed_variety_yield_delta_empty_handling():
    extractor = SeedVarietyYieldDeltaExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
