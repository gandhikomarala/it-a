# Unit Test for FailoverReadinessMetricExtractor_Mininghaultrucks (Autonomous Mining Haul Truck Fleets).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.mining_haul_trucks.failover_readiness_metric import FailoverReadinessMetricExtractor_Mininghaultrucks
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_mining_haul_trucks_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Mininghaultrucks()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_mining_haul_trucks_signal" in res.columns
    assert f"failover_readiness_metric_mining_haul_trucks_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_mining_haul_trucks_signal"].isnull().any()

def test_failover_readiness_metric_mining_haul_trucks_empty():
    extractor = FailoverReadinessMetricExtractor_Mininghaultrucks()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
