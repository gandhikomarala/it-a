# Unit Test for FailoverReadinessMetricExtractor_Industrialmetrologyct (Industrial X-Ray Computed Tomography).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.industrial_metrology_ct.failover_readiness_metric import FailoverReadinessMetricExtractor_Industrialmetrologyct
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_industrial_metrology_ct_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Industrialmetrologyct()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_industrial_metrology_ct_signal" in res.columns
    assert f"failover_readiness_metric_industrial_metrology_ct_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_industrial_metrology_ct_signal"].isnull().any()

def test_failover_readiness_metric_industrial_metrology_ct_empty():
    extractor = FailoverReadinessMetricExtractor_Industrialmetrologyct()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
