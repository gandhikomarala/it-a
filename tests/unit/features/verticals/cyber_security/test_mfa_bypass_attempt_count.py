# Unit Test for MFABypassAttemptCountExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.mfa_bypass_attempt_count import MFABypassAttemptCountExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_mfa_bypass_attempt_count_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MFABypassAttemptCountExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"mfa_bypass_attempt_count_signal" in res.columns
    assert f"mfa_bypass_attempt_count_risk_score" in res.columns
    assert not res[f"mfa_bypass_attempt_count_signal"].isnull().any()

def test_mfa_bypass_attempt_count_empty_dataframe():
    extractor = MFABypassAttemptCountExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
