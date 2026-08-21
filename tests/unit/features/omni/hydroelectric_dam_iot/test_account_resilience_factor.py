# Unit Test for AccountResilienceFactorExtractor_Hydroelectricdamiot (Hydroelectric Dam Structural Health).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.hydroelectric_dam_iot.account_resilience_factor import AccountResilienceFactorExtractor_Hydroelectricdamiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_hydroelectric_dam_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Hydroelectricdamiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_hydroelectric_dam_iot_signal" in res.columns
    assert f"account_resilience_factor_hydroelectric_dam_iot_risk_score" in res.columns
    assert not res[f"account_resilience_factor_hydroelectric_dam_iot_signal"].isnull().any()

def test_account_resilience_factor_hydroelectric_dam_iot_empty():
    extractor = AccountResilienceFactorExtractor_Hydroelectricdamiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
