# Unit Test for AccountResilienceFactorExtractor_Semiconductorfabyield (Semiconductor 3nm Wafer Fab Yield).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.semiconductor_fab_yield.account_resilience_factor import AccountResilienceFactorExtractor_Semiconductorfabyield
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_semiconductor_fab_yield_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Semiconductorfabyield()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_semiconductor_fab_yield_signal" in res.columns
    assert f"account_resilience_factor_semiconductor_fab_yield_risk_score" in res.columns
    assert not res[f"account_resilience_factor_semiconductor_fab_yield_signal"].isnull().any()

def test_account_resilience_factor_semiconductor_fab_yield_empty():
    extractor = AccountResilienceFactorExtractor_Semiconductorfabyield()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
