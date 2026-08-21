# Comprehensive Unit Test for DesktopActiveDaysRatioExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.desktop_active_days_ratio import DesktopActiveDaysRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_desktop_active_days_ratio_instantiation():
    extractor = DesktopActiveDaysRatioExtractor()
    assert extractor.prefix == "desktop_active_days_ratio"

def test_desktop_active_days_ratio_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = DesktopActiveDaysRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("desktop_active_days_ratio_")]
    assert len(expected_cols) > 0

def test_desktop_active_days_ratio_transform_empty():
    extractor = DesktopActiveDaysRatioExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
