# Comprehensive Unit Test for PTOBalanceExhaustionExtractor (HRTech & People Analytics SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.hrtech_talent.pto_balance_exhaustion_rate import PTOBalanceExhaustionExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_pto_balance_exhaustion_rate_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PTOBalanceExhaustionExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"pto_balance_exhaustion_rate_signal" in res.columns
    assert f"pto_balance_exhaustion_rate_risk_score" in res.columns
    assert not res[f"pto_balance_exhaustion_rate_signal"].isnull().any()

def test_pto_balance_exhaustion_rate_empty():
    extractor = PTOBalanceExhaustionExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
