# Comprehensive Unit Test for BattlepassProgressionLagExtractor (Gaming & Interactive Entertainment).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.gaming_media.battlepass_progression_lag import BattlepassProgressionLagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_battlepass_progression_lag_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BattlepassProgressionLagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"battlepass_progression_lag_signal" in res.columns
    assert f"battlepass_progression_lag_risk_score" in res.columns
    assert not res[f"battlepass_progression_lag_signal"].isnull().any()

def test_battlepass_progression_lag_empty():
    extractor = BattlepassProgressionLagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
