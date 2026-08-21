# Unit Test for FailoverReadinessMetricExtractor_Subseapipelinerov (Subsea Oil Pipeline & Deepsea ROV).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.subsea_pipeline_rov.failover_readiness_metric import FailoverReadinessMetricExtractor_Subseapipelinerov
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_failover_readiness_metric_subsea_pipeline_rov_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FailoverReadinessMetricExtractor_Subseapipelinerov()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"failover_readiness_metric_subsea_pipeline_rov_signal" in res.columns
    assert f"failover_readiness_metric_subsea_pipeline_rov_risk_score" in res.columns
    assert not res[f"failover_readiness_metric_subsea_pipeline_rov_signal"].isnull().any()

def test_failover_readiness_metric_subsea_pipeline_rov_empty():
    extractor = FailoverReadinessMetricExtractor_Subseapipelinerov()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
