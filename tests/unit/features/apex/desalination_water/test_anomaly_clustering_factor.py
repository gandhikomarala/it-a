# Unit Test for AnomalyClusteringFactorExtractor_Desalinationwater (Reverse Osmosis Sea Water Desalination).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.desalination_water.anomaly_clustering_factor import AnomalyClusteringFactorExtractor_Desalinationwater
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_anomaly_clustering_factor_desalination_water_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AnomalyClusteringFactorExtractor_Desalinationwater()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"anomaly_clustering_factor_desalination_water_signal" in res.columns
    assert f"anomaly_clustering_factor_desalination_water_risk_score" in res.columns
    assert not res[f"anomaly_clustering_factor_desalination_water_signal"].isnull().any()

def test_anomaly_clustering_factor_desalination_water_empty():
    extractor = AnomalyClusteringFactorExtractor_Desalinationwater()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
