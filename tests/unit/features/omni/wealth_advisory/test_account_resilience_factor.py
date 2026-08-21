# Unit Test for AccountResilienceFactorExtractor_Wealthadvisory (Private Wealth Advisory & Estate Planning).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.wealth_advisory.account_resilience_factor import AccountResilienceFactorExtractor_Wealthadvisory
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_wealth_advisory_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Wealthadvisory()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_wealth_advisory_signal" in res.columns
    assert f"account_resilience_factor_wealth_advisory_risk_score" in res.columns
    assert not res[f"account_resilience_factor_wealth_advisory_signal"].isnull().any()

def test_account_resilience_factor_wealth_advisory_empty():
    extractor = AccountResilienceFactorExtractor_Wealthadvisory()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
