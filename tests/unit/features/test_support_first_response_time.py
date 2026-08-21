# Comprehensive Unit Test for SupportFirstResponseTimeExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.support_first_response_time import SupportFirstResponseTimeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_support_first_response_time_instantiation():
    extractor = SupportFirstResponseTimeExtractor()
    assert extractor.prefix == "support_first_response_time"

def test_support_first_response_time_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = SupportFirstResponseTimeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("support_first_response_time_")]
    assert len(expected_cols) > 0

def test_support_first_response_time_transform_empty():
    extractor = SupportFirstResponseTimeExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
