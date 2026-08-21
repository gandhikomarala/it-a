# Comprehensive Unit Test for AuthorFollowEngagementRateExtractor (Media, OTT & Digital Publishing).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.media_streaming.author_follow_engagement_rate import AuthorFollowEngagementRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_author_follow_engagement_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AuthorFollowEngagementRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"author_follow_engagement_rate_signal" in res.columns
    assert f"author_follow_engagement_rate_risk_score" in res.columns
    assert not res[f"author_follow_engagement_rate_signal"].isnull().any()

def test_author_follow_engagement_rate_empty_handling():
    extractor = AuthorFollowEngagementRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
