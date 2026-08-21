# Comprehensive Unit Test for WinLossTiltExtractor (Gaming & Interactive Entertainment).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.gaming_media.multiplayer_win_loss_tilt import WinLossTiltExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_multiplayer_win_loss_tilt_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = WinLossTiltExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"multiplayer_win_loss_tilt_signal" in res.columns
    assert f"multiplayer_win_loss_tilt_risk_score" in res.columns
    assert not res[f"multiplayer_win_loss_tilt_signal"].isnull().any()

def test_multiplayer_win_loss_tilt_empty():
    extractor = WinLossTiltExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
