# Unit Test for FailoverReadinessMetricExtractor_Quantumcloud (Quantum Computing Cloud Infrastructure).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.quantum_cloud.failover_readiness_metric import FailoverReadinessMetricExtractor_Quantumcloud
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_quantum_cloud_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Quantumcloud()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_quantum_cloud_signal" in res.columns
    assert f"failover_readiness_metric_quantum_cloud_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_quantum_cloud_signal"].isnull().any()

def test_failover_readiness_metric_quantum_cloud_empty():
    extractor = FailoverReadinessMetricExtractor_Quantumcloud()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
