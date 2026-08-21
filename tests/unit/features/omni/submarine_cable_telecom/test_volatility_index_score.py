# Unit Test for VolatilityIndexScoreExtractor_Submarinecabletelecom (Submarine Fiber Cable Capacity).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.submarine_cable_telecom.volatility_index_score import VolatilityIndexScoreExtractor_Submarinecabletelecom
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_submarine_cable_telecom_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Submarinecabletelecom()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_submarine_cable_telecom_signal" in res.columns
    assert f"volatility_index_score_submarine_cable_telecom_risk_score" in res.columns
    assert not res[f"volatility_index_score_submarine_cable_telecom_signal"].isnull().any()

def test_volatility_index_score_submarine_cable_telecom_empty():
    extractor = VolatilityIndexScoreExtractor_Submarinecabletelecom()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
