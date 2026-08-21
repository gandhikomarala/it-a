# Unit Test for AccountResilienceFactorExtractor_Corporatecomplianceethics (Enterprise Ethics Hotline & Whistleblower).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.corporate_compliance_ethics.account_resilience_factor import AccountResilienceFactorExtractor_Corporatecomplianceethics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_corporate_compliance_ethics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Corporatecomplianceethics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_corporate_compliance_ethics_signal" in res.columns
    assert f"account_resilience_factor_corporate_compliance_ethics_risk_score" in res.columns
    assert not res[f"account_resilience_factor_corporate_compliance_ethics_signal"].isnull().any()

def test_account_resilience_factor_corporate_compliance_ethics_empty():
    extractor = AccountResilienceFactorExtractor_Corporatecomplianceethics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
