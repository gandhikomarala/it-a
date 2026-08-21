# Unit Test for AccountBalanceDepletionVelocity (fintech).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.fintech.account_balance_depletion_velocity import AccountBalanceDepletionVelocity
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_balance_depletion_velocity_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountBalanceDepletionVelocity()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_balance_depletion_velocity_signal" in res.columns
    assert f"account_balance_depletion_velocity_risk_index" in res.columns
    assert not res[f"account_balance_depletion_velocity_signal"].isnull().any()

def test_account_balance_depletion_velocity_empty_handling():
    extractor = AccountBalanceDepletionVelocity()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
