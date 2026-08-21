# Unit Test for SecurityPolicyExceptionRequestsExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.security_policy_exception_requests import SecurityPolicyExceptionRequestsExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_security_policy_exception_requests_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SecurityPolicyExceptionRequestsExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"security_policy_exception_requests_signal" in res.columns
    assert f"security_policy_exception_requests_risk_score" in res.columns
    assert not res[f"security_policy_exception_requests_signal"].isnull().any()

def test_security_policy_exception_requests_empty_dataframe():
    extractor = SecurityPolicyExceptionRequestsExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
