# Comprehensive Unit Test for BingeCompletionRateSeriesExtractor (Media, OTT & Digital Publishing).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.media_streaming.binge_completion_rate_series import BingeCompletionRateSeriesExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_binge_completion_rate_series_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BingeCompletionRateSeriesExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"binge_completion_rate_series_signal" in res.columns
    assert f"binge_completion_rate_series_risk_score" in res.columns
    assert not res[f"binge_completion_rate_series_signal"].isnull().any()

def test_binge_completion_rate_series_empty_handling():
    extractor = BingeCompletionRateSeriesExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
