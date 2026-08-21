# Unit Test for EngagementMomentumExtractor_Behavioralmentalhealth (Behavioral & Mental Health Telehealth).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.behavioral_mental_health.engagement_momentum import EngagementMomentumExtractor_Behavioralmentalhealth
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_engagement_momentum_behavioral_mental_health_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EngagementMomentumExtractor_Behavioralmentalhealth()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"engagement_momentum_behavioral_mental_health_signal" in res.columns
    assert f"engagement_momentum_behavioral_mental_health_risk_score" in res.columns
    assert not res[f"engagement_momentum_behavioral_mental_health_signal"].isnull().any()

def test_engagement_momentum_behavioral_mental_health_empty():
    extractor = EngagementMomentumExtractor_Behavioralmentalhealth()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
