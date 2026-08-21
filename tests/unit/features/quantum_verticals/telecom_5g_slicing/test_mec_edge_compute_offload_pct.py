# Comprehensive Unit Test for MECEdgeComputeOffloadExtractor (Telecom 5G Network Slicing).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.telecom_5g_slicing.mec_edge_compute_offload_pct import MECEdgeComputeOffloadExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_mec_edge_compute_offload_pct_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MECEdgeComputeOffloadExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"mec_edge_compute_offload_pct_signal" in res.columns
    assert f"mec_edge_compute_offload_pct_risk_score" in res.columns
    assert not res[f"mec_edge_compute_offload_pct_signal"].isnull().any()

def test_mec_edge_compute_offload_pct_empty_handling():
    extractor = MECEdgeComputeOffloadExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
