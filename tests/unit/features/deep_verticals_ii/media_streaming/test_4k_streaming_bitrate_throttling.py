# Comprehensive Unit Test for StreamingBitrateThrottlingExtractor (Media, OTT & Digital Publishing).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.media_streaming.4k_streaming_bitrate_throttling import StreamingBitrateThrottlingExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_4k_streaming_bitrate_throttling_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = StreamingBitrateThrottlingExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"4k_streaming_bitrate_throttling_signal" in res.columns
    assert f"4k_streaming_bitrate_throttling_risk_score" in res.columns
    assert not res[f"4k_streaming_bitrate_throttling_signal"].isnull().any()

def test_4k_streaming_bitrate_throttling_empty_handling():
    extractor = StreamingBitrateThrottlingExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
