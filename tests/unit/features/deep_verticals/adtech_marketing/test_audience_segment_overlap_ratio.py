# Comprehensive Unit Test for AudienceOverlapRatioExtractor (AdTech & Growth Marketing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.adtech_marketing.audience_segment_overlap_ratio import AudienceOverlapRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_audience_segment_overlap_ratio_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AudienceOverlapRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"audience_segment_overlap_ratio_signal" in res.columns
    assert f"audience_segment_overlap_ratio_risk_score" in res.columns
    assert not res[f"audience_segment_overlap_ratio_signal"].isnull().any()

def test_audience_segment_overlap_ratio_empty():
    extractor = AudienceOverlapRatioExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
