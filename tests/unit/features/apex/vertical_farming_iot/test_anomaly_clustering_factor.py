# Unit Test for AnomalyClusteringFactorExtractor_Verticalfarmingiot (Controlled Environment Vertical Agriculture).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.vertical_farming_iot.anomaly_clustering_factor import AnomalyClusteringFactorExtractor_Verticalfarmingiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_anomaly_clustering_factor_vertical_farming_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AnomalyClusteringFactorExtractor_Verticalfarmingiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"anomaly_clustering_factor_vertical_farming_iot_signal" in res.columns
    assert f"anomaly_clustering_factor_vertical_farming_iot_risk_score" in res.columns
    assert not res[f"anomaly_clustering_factor_vertical_farming_iot_signal"].isnull().any()

def test_anomaly_clustering_factor_vertical_farming_iot_empty():
    extractor = AnomalyClusteringFactorExtractor_Verticalfarmingiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
