# Unit Test for AccountResilienceFactorExtractor_Reinsurancecatastrophe (Catastrophe Reinsurance Modeling).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.reinsurance_catastrophe.account_resilience_factor import AccountResilienceFactorExtractor_Reinsurancecatastrophe
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_account_resilience_factor_reinsurance_catastrophe_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AccountResilienceFactorExtractor_Reinsurancecatastrophe()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"account_resilience_factor_reinsurance_catastrophe_signal" in res.columns
    assert f"account_resilience_factor_reinsurance_catastrophe_risk_score" in res.columns
    assert not res[f"account_resilience_factor_reinsurance_catastrophe_signal"].isnull().any()

def test_account_resilience_factor_reinsurance_catastrophe_empty():
    extractor = AccountResilienceFactorExtractor_Reinsurancecatastrophe()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
