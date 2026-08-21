# Comprehensive Unit Test for RemoteStartAppActiveStreakExtractor (Automotive & Connected Fleet).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.automotive_telematics.remote_start_app_active_streak import RemoteStartAppActiveStreakExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_remote_start_app_active_streak_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RemoteStartAppActiveStreakExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"remote_start_app_active_streak_signal" in res.columns
    assert f"remote_start_app_active_streak_risk_score" in res.columns
    assert not res[f"remote_start_app_active_streak_signal"].isnull().any()

def test_remote_start_app_active_streak_empty_handling():
    extractor = RemoteStartAppActiveStreakExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
