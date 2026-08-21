# Comprehensive Unit Test for SliceBandwidthThrottlingExtractor (Telecom 5G Network Slicing).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.telecom_5g_slicing.slice_bandwidth_throttling_events import SliceBandwidthThrottlingExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_slice_bandwidth_throttling_events_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SliceBandwidthThrottlingExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"slice_bandwidth_throttling_events_signal" in res.columns
    assert f"slice_bandwidth_throttling_events_risk_score" in res.columns
    assert not res[f"slice_bandwidth_throttling_events_signal"].isnull().any()

def test_slice_bandwidth_throttling_events_empty_handling():
    extractor = SliceBandwidthThrottlingExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
