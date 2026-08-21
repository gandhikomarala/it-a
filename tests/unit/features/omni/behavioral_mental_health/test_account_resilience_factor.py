# Unit Test for AccountResilienceFactorExtractor_Behavioralmentalhealth (Behavioral & Mental Health Telehealth).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.behavioral_mental_health.account_resilience_factor import AccountResilienceFactorExtractor_Behavioralmentalhealth
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_behavioral_mental_health_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Behavioralmentalhealth()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_behavioral_mental_health_signal" in res.columns
    assert f"account_resilience_factor_behavioral_mental_health_risk_score" in res.columns
    assert not res[f"account_resilience_factor_behavioral_mental_health_signal"].isnull().any()

def test_account_resilience_factor_behavioral_mental_health_empty():
    extractor = AccountResilienceFactorExtractor_Behavioralmentalhealth()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
