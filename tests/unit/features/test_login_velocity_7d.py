# Comprehensive Unit Test for LoginVelocity7dExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.login_velocity_7d import LoginVelocity7dExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_login_velocity_7d_instantiation():
    extractor = LoginVelocity7dExtractor()
    assert extractor.prefix == "login_velocity_7d"

def test_login_velocity_7d_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = LoginVelocity7dExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("login_velocity_7d_")]
    assert len(expected_cols) > 0

def test_login_velocity_7d_transform_empty():
    extractor = LoginVelocity7dExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
