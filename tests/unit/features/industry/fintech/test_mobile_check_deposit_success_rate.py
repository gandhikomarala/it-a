# Unit Test for MobileDepositSuccessRate (fintech).
import pytest
import numpy as np
import pandas as pd
from ml.features.industry.fintech.mobile_check_deposit_success_rate import MobileDepositSuccessRate
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_mobile_check_deposit_success_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MobileDepositSuccessRate()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"mobile_check_deposit_success_rate_signal" in res.columns
    assert f"mobile_check_deposit_success_rate_risk_index" in res.columns
    assert not res[f"mobile_check_deposit_success_rate_signal"].isnull().any()

def test_mobile_check_deposit_success_rate_empty_handling():
    extractor = MobileDepositSuccessRate()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
