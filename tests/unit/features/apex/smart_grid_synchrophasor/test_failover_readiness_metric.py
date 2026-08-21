# Unit Test for FailoverReadinessMetricExtractor_Smartgridsynchrophasor (Smart Grid PMU Synchrophasor Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_grid_synchrophasor.failover_readiness_metric import FailoverReadinessMetricExtractor_Smartgridsynchrophasor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_smart_grid_synchrophasor_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Smartgridsynchrophasor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_smart_grid_synchrophasor_signal" in res.columns
    assert f"failover_readiness_metric_smart_grid_synchrophasor_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_smart_grid_synchrophasor_signal"].isnull().any()

def test_failover_readiness_metric_smart_grid_synchrophasor_empty():
    extractor = FailoverReadinessMetricExtractor_Smartgridsynchrophasor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
