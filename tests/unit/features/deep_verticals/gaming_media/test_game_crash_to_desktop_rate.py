# Comprehensive Unit Test for GameCrashRateExtractor (Gaming & Interactive Entertainment).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.gaming_media.game_crash_to_desktop_rate import GameCrashRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_game_crash_to_desktop_rate_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = GameCrashRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"game_crash_to_desktop_rate_signal" in res.columns
    assert f"game_crash_to_desktop_rate_risk_score" in res.columns
    assert not res[f"game_crash_to_desktop_rate_signal"].isnull().any()

def test_game_crash_to_desktop_rate_empty():
    extractor = GameCrashRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
