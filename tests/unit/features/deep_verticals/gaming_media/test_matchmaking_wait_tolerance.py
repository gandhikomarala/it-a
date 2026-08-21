# Comprehensive Unit Test for MatchmakingWaitToleranceExtractor (Gaming & Interactive Entertainment).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.gaming_media.matchmaking_wait_tolerance import MatchmakingWaitToleranceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_matchmaking_wait_tolerance_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MatchmakingWaitToleranceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"matchmaking_wait_tolerance_signal" in res.columns
    assert f"matchmaking_wait_tolerance_risk_score" in res.columns
    assert not res[f"matchmaking_wait_tolerance_signal"].isnull().any()

def test_matchmaking_wait_tolerance_empty():
    extractor = MatchmakingWaitToleranceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
