# Unit Test for VolatilityIndexScoreExtractor_Behavioralmentalhealth (Behavioral & Mental Health Telehealth).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.behavioral_mental_health.volatility_index_score import VolatilityIndexScoreExtractor_Behavioralmentalhealth
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_volatility_index_score_behavioral_mental_health_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = VolatilityIndexScoreExtractor_Behavioralmentalhealth()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"volatility_index_score_behavioral_mental_health_signal" in res.columns
    assert f"volatility_index_score_behavioral_mental_health_risk_score" in res.columns
    assert not res[f"volatility_index_score_behavioral_mental_health_signal"].isnull().any()

def test_volatility_index_score_behavioral_mental_health_empty():
    extractor = VolatilityIndexScoreExtractor_Behavioralmentalhealth()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
