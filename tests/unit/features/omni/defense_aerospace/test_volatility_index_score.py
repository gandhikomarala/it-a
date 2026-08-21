# Unit Test for VolatilityIndexScoreExtractor_Defenseaerospace (Defense & Aerospace Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.defense_aerospace.volatility_index_score import VolatilityIndexScoreExtractor_Defenseaerospace
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_defense_aerospace_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Defenseaerospace()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_defense_aerospace_signal" in res.columns
    assert f"volatility_index_score_defense_aerospace_risk_score" in res.columns
    assert not res[f"volatility_index_score_defense_aerospace_signal"].isnull().any()

def test_volatility_index_score_defense_aerospace_empty():
    extractor = VolatilityIndexScoreExtractor_Defenseaerospace()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
