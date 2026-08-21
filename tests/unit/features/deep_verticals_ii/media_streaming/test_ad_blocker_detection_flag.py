# Comprehensive Unit Test for AdBlockerDetectionFlagExtractor (Media, OTT & Digital Publishing).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.media_streaming.ad_blocker_detection_flag import AdBlockerDetectionFlagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ad_blocker_detection_flag_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AdBlockerDetectionFlagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ad_blocker_detection_flag_signal" in res.columns
    assert f"ad_blocker_detection_flag_risk_score" in res.columns
    assert not res[f"ad_blocker_detection_flag_signal"].isnull().any()

def test_ad_blocker_detection_flag_empty_handling():
    extractor = AdBlockerDetectionFlagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
