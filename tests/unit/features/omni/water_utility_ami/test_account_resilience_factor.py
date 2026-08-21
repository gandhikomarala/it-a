# Unit Test for AccountResilienceFactorExtractor_Waterutilityami (Municipal Smart Water AMI Network).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.water_utility_ami.account_resilience_factor import AccountResilienceFactorExtractor_Waterutilityami
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_water_utility_ami_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Waterutilityami()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_water_utility_ami_signal" in res.columns
    assert f"account_resilience_factor_water_utility_ami_risk_score" in res.columns
    assert not res[f"account_resilience_factor_water_utility_ami_signal"].isnull().any()

def test_account_resilience_factor_water_utility_ami_empty():
    extractor = AccountResilienceFactorExtractor_Waterutilityami()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
