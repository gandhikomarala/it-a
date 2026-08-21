# Comprehensive Unit Test for QuarterEndActivitySpikeExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.quarter_end_activity_spike import QuarterEndActivitySpikeExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quarter_end_activity_spike_instantiation():
    extractor = QuarterEndActivitySpikeExtractor()
    assert extractor.prefix == "quarter_end_activity_spike"

def test_quarter_end_activity_spike_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = QuarterEndActivitySpikeExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("quarter_end_activity_spike_")]
    assert len(expected_cols) > 0

def test_quarter_end_activity_spike_transform_empty():
    extractor = QuarterEndActivitySpikeExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
