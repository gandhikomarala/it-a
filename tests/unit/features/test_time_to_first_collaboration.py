# Comprehensive Unit Test for TimeToFirstCollaborationExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.time_to_first_collaboration import TimeToFirstCollaborationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_time_to_first_collaboration_instantiation():
    extractor = TimeToFirstCollaborationExtractor()
    assert extractor.prefix == "time_to_first_collaboration"

def test_time_to_first_collaboration_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = TimeToFirstCollaborationExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("time_to_first_collaboration_")]
    assert len(expected_cols) > 0

def test_time_to_first_collaboration_transform_empty():
    extractor = TimeToFirstCollaborationExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
