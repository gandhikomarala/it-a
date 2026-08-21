# Unit Test for AnomalyClusteringFactorExtractor_Geothermalenergydeep (Enhanced Geothermal Deep Drilling Systems).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.geothermal_energy_deep.anomaly_clustering_factor import AnomalyClusteringFactorExtractor_Geothermalenergydeep
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_anomaly_clustering_factor_geothermal_energy_deep_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AnomalyClusteringFactorExtractor_Geothermalenergydeep()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"anomaly_clustering_factor_geothermal_energy_deep_signal" in res.columns
    assert f"anomaly_clustering_factor_geothermal_energy_deep_risk_score" in res.columns
    assert not res[f"anomaly_clustering_factor_geothermal_energy_deep_signal"].isnull().any()

def test_anomaly_clustering_factor_geothermal_energy_deep_empty():
    extractor = AnomalyClusteringFactorExtractor_Geothermalenergydeep()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
