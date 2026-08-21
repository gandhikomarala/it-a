# Comprehensive Unit Test for AdminActivityRatioExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.admin_activity_ratio import AdminActivityRatioExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_admin_activity_ratio_instantiation():
    extractor = AdminActivityRatioExtractor()
    assert extractor.prefix == "admin_activity_ratio"

def test_admin_activity_ratio_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = AdminActivityRatioExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("admin_activity_ratio_")]
    assert len(expected_cols) > 0

def test_admin_activity_ratio_transform_empty():
    extractor = AdminActivityRatioExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
