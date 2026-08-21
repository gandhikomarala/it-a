# Comprehensive Unit Test for FeatureUsageDepthExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.feature_usage_depth import FeatureUsageDepthExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_usage_depth_instantiation():
    extractor = FeatureUsageDepthExtractor()
    assert extractor.prefix == "feature_usage_depth"

def test_feature_usage_depth_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = FeatureUsageDepthExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("feature_usage_depth_")]
    assert len(expected_cols) > 0

def test_feature_usage_depth_transform_empty():
    extractor = FeatureUsageDepthExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
