# Unit Test for FailoverReadinessMetricExtractor_Timbersmartforestry (Autonomous Precision Forestry Harvesters).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.timber_smart_forestry.failover_readiness_metric import FailoverReadinessMetricExtractor_Timbersmartforestry
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_timber_smart_forestry_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Timbersmartforestry()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_timber_smart_forestry_signal" in res.columns
    assert f"failover_readiness_metric_timber_smart_forestry_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_timber_smart_forestry_signal"].isnull().any()

def test_failover_readiness_metric_timber_smart_forestry_empty():
    extractor = FailoverReadinessMetricExtractor_Timbersmartforestry()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
