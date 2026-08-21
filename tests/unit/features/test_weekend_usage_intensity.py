# Comprehensive Unit Test for WeekendUsageIntensityExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.weekend_usage_intensity import WeekendUsageIntensityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_weekend_usage_intensity_instantiation():
    extractor = WeekendUsageIntensityExtractor()
    assert extractor.prefix == "weekend_usage_intensity"

def test_weekend_usage_intensity_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = WeekendUsageIntensityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("weekend_usage_intensity_")]
    assert len(expected_cols) > 0

def test_weekend_usage_intensity_transform_empty():
    extractor = WeekendUsageIntensityExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
