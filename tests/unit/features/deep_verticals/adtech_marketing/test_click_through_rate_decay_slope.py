# Comprehensive Unit Test for CTRDecaySlopeExtractor (AdTech & Growth Marketing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.adtech_marketing.click_through_rate_decay_slope import CTRDecaySlopeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_click_through_rate_decay_slope_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CTRDecaySlopeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"click_through_rate_decay_slope_signal" in res.columns
    assert f"click_through_rate_decay_slope_risk_score" in res.columns
    assert not res[f"click_through_rate_decay_slope_signal"].isnull().any()

def test_click_through_rate_decay_slope_empty():
    extractor = CTRDecaySlopeExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
