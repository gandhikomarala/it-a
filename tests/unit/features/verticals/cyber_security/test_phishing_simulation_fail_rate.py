# Unit Test for PhishingSimFailRateExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.phishing_simulation_fail_rate import PhishingSimFailRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_phishing_simulation_fail_rate_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PhishingSimFailRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"phishing_simulation_fail_rate_signal" in res.columns
    assert f"phishing_simulation_fail_rate_risk_score" in res.columns
    assert not res[f"phishing_simulation_fail_rate_signal"].isnull().any()

def test_phishing_simulation_fail_rate_empty_dataframe():
    extractor = PhishingSimFailRateExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
