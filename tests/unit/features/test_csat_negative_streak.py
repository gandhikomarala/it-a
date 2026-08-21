# Comprehensive Unit Test for CSATNegativeStreakExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.csat_negative_streak import CSATNegativeStreakExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_csat_negative_streak_instantiation():
    extractor = CSATNegativeStreakExtractor()
    assert extractor.prefix == "csat_negative_streak"

def test_csat_negative_streak_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = CSATNegativeStreakExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("csat_negative_streak_")]
    assert len(expected_cols) > 0

def test_csat_negative_streak_transform_empty():
    extractor = CSATNegativeStreakExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
