# Unit Test for VolatilityIndexScoreExtractor_Cruiselineshospitality (Cruise Line Passenger Lifetime Value).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cruise_lines_hospitality.volatility_index_score import VolatilityIndexScoreExtractor_Cruiselineshospitality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_cruise_lines_hospitality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Cruiselineshospitality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_cruise_lines_hospitality_signal" in res.columns
    assert f"volatility_index_score_cruise_lines_hospitality_risk_score" in res.columns
    assert not res[f"volatility_index_score_cruise_lines_hospitality_signal"].isnull().any()

def test_volatility_index_score_cruise_lines_hospitality_empty():
    extractor = VolatilityIndexScoreExtractor_Cruiselineshospitality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
