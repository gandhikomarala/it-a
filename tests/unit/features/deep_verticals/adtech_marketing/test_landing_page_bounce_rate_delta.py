# Comprehensive Unit Test for LandingPageBounceDeltaExtractor (AdTech & Growth Marketing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.adtech_marketing.landing_page_bounce_rate_delta import LandingPageBounceDeltaExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_landing_page_bounce_rate_delta_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LandingPageBounceDeltaExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"landing_page_bounce_rate_delta_signal" in res.columns
    assert f"landing_page_bounce_rate_delta_risk_score" in res.columns
    assert not res[f"landing_page_bounce_rate_delta_signal"].isnull().any()

def test_landing_page_bounce_rate_delta_empty():
    extractor = LandingPageBounceDeltaExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
