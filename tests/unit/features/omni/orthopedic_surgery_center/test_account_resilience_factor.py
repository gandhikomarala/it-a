# Unit Test for AccountResilienceFactorExtractor_Orthopedicsurgerycenter (Ambulatory Surgical Center Operations).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.orthopedic_surgery_center.account_resilience_factor import AccountResilienceFactorExtractor_Orthopedicsurgerycenter
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_orthopedic_surgery_center_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Orthopedicsurgerycenter()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_orthopedic_surgery_center_signal" in res.columns
    assert f"account_resilience_factor_orthopedic_surgery_center_risk_score" in res.columns
    assert not res[f"account_resilience_factor_orthopedic_surgery_center_signal"].isnull().any()

def test_account_resilience_factor_orthopedic_surgery_center_empty():
    extractor = AccountResilienceFactorExtractor_Orthopedicsurgerycenter()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
