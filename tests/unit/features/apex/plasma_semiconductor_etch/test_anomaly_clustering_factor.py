# Unit Test for AnomalyClusteringFactorExtractor_Plasmasemiconductoretch (Atomic Layer Semiconductor Plasma Etch).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.plasma_semiconductor_etch.anomaly_clustering_factor import AnomalyClusteringFactorExtractor_Plasmasemiconductoretch
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_anomaly_clustering_factor_plasma_semiconductor_etch_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AnomalyClusteringFactorExtractor_Plasmasemiconductoretch()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"anomaly_clustering_factor_plasma_semiconductor_etch_signal" in res.columns
    assert f"anomaly_clustering_factor_plasma_semiconductor_etch_risk_score" in res.columns
    assert not res[f"anomaly_clustering_factor_plasma_semiconductor_etch_signal"].isnull().any()

def test_anomaly_clustering_factor_plasma_semiconductor_etch_empty():
    extractor = AnomalyClusteringFactorExtractor_Plasmasemiconductoretch()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
