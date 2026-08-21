# Comprehensive Unit Test for CoreAMFAuthLatencyExtractor (Telecom 5G Network Slicing).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.telecom_5g_slicing.core_amf_authentication_latency import CoreAMFAuthLatencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_core_amf_authentication_latency_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CoreAMFAuthLatencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"core_amf_authentication_latency_signal" in res.columns
    assert f"core_amf_authentication_latency_risk_score" in res.columns
    assert not res[f"core_amf_authentication_latency_signal"].isnull().any()

def test_core_amf_authentication_latency_empty_handling():
    extractor = CoreAMFAuthLatencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
