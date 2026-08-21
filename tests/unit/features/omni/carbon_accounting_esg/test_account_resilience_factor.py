# Unit Test for AccountResilienceFactorExtractor_Carbonaccountingesg (Enterprise Scope 1-2-3 Carbon Accounting).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.carbon_accounting_esg.account_resilience_factor import AccountResilienceFactorExtractor_Carbonaccountingesg
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_carbon_accounting_esg_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Carbonaccountingesg()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_carbon_accounting_esg_signal" in res.columns
    assert f"account_resilience_factor_carbon_accounting_esg_risk_score" in res.columns
    assert not res[f"account_resilience_factor_carbon_accounting_esg_signal"].isnull().any()

def test_account_resilience_factor_carbon_accounting_esg_empty():
    extractor = AccountResilienceFactorExtractor_Carbonaccountingesg()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
