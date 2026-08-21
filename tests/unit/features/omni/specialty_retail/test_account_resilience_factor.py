# Unit Test for AccountResilienceFactorExtractor_Specialtyretail (Specialty Retail Omnichannel Inventory).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.specialty_retail.account_resilience_factor import AccountResilienceFactorExtractor_Specialtyretail
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_specialty_retail_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Specialtyretail()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_specialty_retail_signal" in res.columns
    assert f"account_resilience_factor_specialty_retail_risk_score" in res.columns
    assert not res[f"account_resilience_factor_specialty_retail_signal"].isnull().any()

def test_account_resilience_factor_specialty_retail_empty():
    extractor = AccountResilienceFactorExtractor_Specialtyretail()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
