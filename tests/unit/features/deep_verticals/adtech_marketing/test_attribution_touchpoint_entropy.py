# Comprehensive Unit Test for AttributionTouchpointEntropyExtractor (AdTech & Growth Marketing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.adtech_marketing.attribution_touchpoint_entropy import AttributionTouchpointEntropyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_attribution_touchpoint_entropy_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AttributionTouchpointEntropyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"attribution_touchpoint_entropy_signal" in res.columns
    assert f"attribution_touchpoint_entropy_risk_score" in res.columns
    assert not res[f"attribution_touchpoint_entropy_signal"].isnull().any()

def test_attribution_touchpoint_entropy_empty():
    extractor = AttributionTouchpointEntropyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
