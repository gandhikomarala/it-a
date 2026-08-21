# Unit Test for AnomalyClusteringFactorExtractor_Subseapipelinerov (Subsea Oil Pipeline & Deepsea ROV).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.subsea_pipeline_rov.anomaly_clustering_factor import AnomalyClusteringFactorExtractor_Subseapipelinerov
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_anomaly_clustering_factor_subsea_pipeline_rov_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AnomalyClusteringFactorExtractor_Subseapipelinerov()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"anomaly_clustering_factor_subsea_pipeline_rov_signal" in res.columns
    assert f"anomaly_clustering_factor_subsea_pipeline_rov_risk_score" in res.columns
    assert not res[f"anomaly_clustering_factor_subsea_pipeline_rov_signal"].isnull().any()

def test_anomaly_clustering_factor_subsea_pipeline_rov_empty():
    extractor = AnomalyClusteringFactorExtractor_Subseapipelinerov()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
