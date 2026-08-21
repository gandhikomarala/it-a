# Comprehensive Unit Test for FailedLoginBurstFlagExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.failed_login_burst_flag import FailedLoginBurstFlagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failed_login_burst_flag_instantiation():
    extractor = FailedLoginBurstFlagExtractor()
    assert extractor.prefix == "failed_login_burst_flag"

def test_failed_login_burst_flag_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = FailedLoginBurstFlagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("failed_login_burst_flag_")]
    assert len(expected_cols) > 0

def test_failed_login_burst_flag_transform_empty():
    extractor = FailedLoginBurstFlagExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
