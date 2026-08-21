# Unit Test for DecayGradientScoreExtractor_Cruiselineshospitality (Cruise Line Passenger Lifetime Value).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cruise_lines_hospitality.decay_gradient_score import DecayGradientScoreExtractor_Cruiselineshospitality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_cruise_lines_hospitality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Cruiselineshospitality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_cruise_lines_hospitality_signal" in res.columns
    assert f"decay_gradient_score_cruise_lines_hospitality_risk_score" in res.columns
    assert not res[f"decay_gradient_score_cruise_lines_hospitality_signal"].isnull().any()

def test_decay_gradient_score_cruise_lines_hospitality_empty():
    extractor = DecayGradientScoreExtractor_Cruiselineshospitality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
