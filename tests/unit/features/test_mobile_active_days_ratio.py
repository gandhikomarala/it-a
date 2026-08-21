# Comprehensive Unit Test for MobileActiveDaysRatioExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.mobile_active_days_ratio import MobileActiveDaysRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_mobile_active_days_ratio_instantiation():
    extractor = MobileActiveDaysRatioExtractor()
    assert extractor.prefix == "mobile_active_days_ratio"

def test_mobile_active_days_ratio_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = MobileActiveDaysRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("mobile_active_days_ratio_")]
    assert len(expected_cols) > 0

def test_mobile_active_days_ratio_transform_empty():
    extractor = MobileActiveDaysRatioExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
