# Unit Test for FailoverReadinessMetricExtractor_Desalinationwater (Reverse Osmosis Sea Water Desalination).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.desalination_water.failover_readiness_metric import FailoverReadinessMetricExtractor_Desalinationwater
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_desalination_water_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Desalinationwater()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_desalination_water_signal" in res.columns
    assert f"failover_readiness_metric_desalination_water_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_desalination_water_signal"].isnull().any()

def test_failover_readiness_metric_desalination_water_empty():
    extractor = FailoverReadinessMetricExtractor_Desalinationwater()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
