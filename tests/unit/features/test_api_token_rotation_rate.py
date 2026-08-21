# Comprehensive Unit Test for APITokenRotationRateExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.api_token_rotation_rate import APITokenRotationRateExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_api_token_rotation_rate_instantiation():
    extractor = APITokenRotationRateExtractor()
    assert extractor.prefix == "api_token_rotation_rate"

def test_api_token_rotation_rate_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = APITokenRotationRateExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("api_token_rotation_rate_")]
    assert len(expected_cols) > 0

def test_api_token_rotation_rate_transform_empty():
    extractor = APITokenRotationRateExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
