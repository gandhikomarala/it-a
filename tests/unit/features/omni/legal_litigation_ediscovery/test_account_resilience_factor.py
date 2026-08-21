# Unit Test for AccountResilienceFactorExtractor_Legallitigationediscovery (Complex Litigation E-Discovery Review).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.legal_litigation_ediscovery.account_resilience_factor import AccountResilienceFactorExtractor_Legallitigationediscovery
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_legal_litigation_ediscovery_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Legallitigationediscovery()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_legal_litigation_ediscovery_signal" in res.columns
    assert f"account_resilience_factor_legal_litigation_ediscovery_risk_score" in res.columns
    assert not res[f"account_resilience_factor_legal_litigation_ediscovery_signal"].isnull().any()

def test_account_resilience_factor_legal_litigation_ediscovery_empty():
    extractor = AccountResilienceFactorExtractor_Legallitigationediscovery()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
