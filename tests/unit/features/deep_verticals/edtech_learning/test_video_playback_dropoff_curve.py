# Comprehensive Unit Test for VideoPlaybackDropoffExtractor (EdTech & Corporate Learning SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.edtech_learning.video_playback_dropoff_curve import VideoPlaybackDropoffExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_video_playback_dropoff_curve_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VideoPlaybackDropoffExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"video_playback_dropoff_curve_signal" in res.columns
    assert f"video_playback_dropoff_curve_risk_score" in res.columns
    assert not res[f"video_playback_dropoff_curve_signal"].isnull().any()

def test_video_playback_dropoff_curve_empty():
    extractor = VideoPlaybackDropoffExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
