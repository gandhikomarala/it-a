# Unit Test for AccountResilienceFactorExtractor_Dentaldsoanalytics (Dental DSO Practice Optimization).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.dental_dso_analytics.account_resilience_factor import AccountResilienceFactorExtractor_Dentaldsoanalytics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_dental_dso_analytics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Dentaldsoanalytics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_dental_dso_analytics_signal" in res.columns
    assert f"account_resilience_factor_dental_dso_analytics_risk_score" in res.columns
    assert not res[f"account_resilience_factor_dental_dso_analytics_signal"].isnull().any()

def test_account_resilience_factor_dental_dso_analytics_empty():
    extractor = AccountResilienceFactorExtractor_Dentaldsoanalytics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
