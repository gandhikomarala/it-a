# Comprehensive Unit Test for SupportResolutionTimeVarianceExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.support_resolution_time_variance import SupportResolutionTimeVarianceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_support_resolution_time_variance_instantiation():
    extractor = SupportResolutionTimeVarianceExtractor()
    assert extractor.prefix == "support_resolution_time_variance"

def test_support_resolution_time_variance_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = SupportResolutionTimeVarianceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("support_resolution_time_variance_")]
    assert len(expected_cols) > 0

def test_support_resolution_time_variance_transform_empty():
    extractor = SupportResolutionTimeVarianceExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
