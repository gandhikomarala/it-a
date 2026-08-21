# Comprehensive Unit Test for NewsletterCTORatioExtractor (Media, OTT & Digital Publishing).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.media_streaming.newsletter_click_to_open_ratio import NewsletterCTORatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_newsletter_click_to_open_ratio_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = NewsletterCTORatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"newsletter_click_to_open_ratio_signal" in res.columns
    assert f"newsletter_click_to_open_ratio_risk_score" in res.columns
    assert not res[f"newsletter_click_to_open_ratio_signal"].isnull().any()

def test_newsletter_click_to_open_ratio_empty_handling():
    extractor = NewsletterCTORatioExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
