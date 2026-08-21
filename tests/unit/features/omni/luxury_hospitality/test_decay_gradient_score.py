# Unit Test for DecayGradientScoreExtractor_Luxuryhospitality (Luxury Resort Concierge Guest Experience).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.luxury_hospitality.decay_gradient_score import DecayGradientScoreExtractor_Luxuryhospitality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_gradient_score_luxury_hospitality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = DecayGradientScoreExtractor_Luxuryhospitality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"decay_gradient_score_luxury_hospitality_signal" in res.columns
    assert f"decay_gradient_score_luxury_hospitality_risk_score" in res.columns
    assert not res[f"decay_gradient_score_luxury_hospitality_signal"].isnull().any()

def test_decay_gradient_score_luxury_hospitality_empty():
    extractor = DecayGradientScoreExtractor_Luxuryhospitality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
