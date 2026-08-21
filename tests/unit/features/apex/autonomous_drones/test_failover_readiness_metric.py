# Unit Test for FailoverReadinessMetricExtractor_Autonomousdrones (Autonomous Commercial Drone Delivery).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.autonomous_drones.failover_readiness_metric import FailoverReadinessMetricExtractor_Autonomousdrones
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_autonomous_drones_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Autonomousdrones()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_autonomous_drones_signal" in res.columns
    assert f"failover_readiness_metric_autonomous_drones_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_autonomous_drones_signal"].isnull().any()

def test_failover_readiness_metric_autonomous_drones_empty():
    extractor = FailoverReadinessMetricExtractor_Autonomousdrones()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
