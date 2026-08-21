# Comprehensive Unit Test for DiscountUtilizationPctExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.discount_utilization_pct import DiscountUtilizationPctExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_discount_utilization_pct_instantiation():
    extractor = DiscountUtilizationPctExtractor()
    assert extractor.prefix == "discount_utilization_pct"

def test_discount_utilization_pct_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = DiscountUtilizationPctExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("discount_utilization_pct_")]
    assert len(expected_cols) > 0

def test_discount_utilization_pct_transform_empty():
    extractor = DiscountUtilizationPctExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
