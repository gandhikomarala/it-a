# Comprehensive Unit Test for SliceLatencySLABreachExtractor (Telecom 5G Network Slicing).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.telecom_5g_slicing.slice_latency_sla_breach_rate import SliceLatencySLABreachExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_slice_latency_sla_breach_rate_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SliceLatencySLABreachExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"slice_latency_sla_breach_rate_signal" in res.columns
    assert f"slice_latency_sla_breach_rate_risk_score" in res.columns
    assert not res[f"slice_latency_sla_breach_rate_signal"].isnull().any()

def test_slice_latency_sla_breach_rate_empty_handling():
    extractor = SliceLatencySLABreachExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
