# Unit Test for FailoverReadinessMetricExtractor_Hyperscaledatacenters (Hyperscale Datacenter Liquid Cooling).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hyperscale_datacenters.failover_readiness_metric import FailoverReadinessMetricExtractor_Hyperscaledatacenters
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_hyperscale_datacenters_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Hyperscaledatacenters()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_hyperscale_datacenters_signal" in res.columns
    assert f"failover_readiness_metric_hyperscale_datacenters_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_hyperscale_datacenters_signal"].isnull().any()

def test_failover_readiness_metric_hyperscale_datacenters_empty():
    extractor = FailoverReadinessMetricExtractor_Hyperscaledatacenters()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
