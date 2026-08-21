# Unit Test for BruteForceAuthAttemptsExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.brute_force_auth_attempts import BruteForceAuthAttemptsExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_brute_force_auth_attempts_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BruteForceAuthAttemptsExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"brute_force_auth_attempts_signal" in res.columns
    assert f"brute_force_auth_attempts_risk_score" in res.columns
    assert not res[f"brute_force_auth_attempts_signal"].isnull().any()

def test_brute_force_auth_attempts_empty_dataframe():
    extractor = BruteForceAuthAttemptsExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
