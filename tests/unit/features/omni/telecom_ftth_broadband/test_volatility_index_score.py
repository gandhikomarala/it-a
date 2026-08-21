# Unit Test for VolatilityIndexScoreExtractor_Telecomftthbroadband (FTTH Fiber Gigabit Broadband Access).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.telecom_ftth_broadband.volatility_index_score import VolatilityIndexScoreExtractor_Telecomftthbroadband
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_telecom_ftth_broadband_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Telecomftthbroadband()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_telecom_ftth_broadband_signal" in res.columns
    assert f"volatility_index_score_telecom_ftth_broadband_risk_score" in res.columns
    assert not res[f"volatility_index_score_telecom_ftth_broadband_signal"].isnull().any()

def test_volatility_index_score_telecom_ftth_broadband_empty():
    extractor = VolatilityIndexScoreExtractor_Telecomftthbroadband()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
