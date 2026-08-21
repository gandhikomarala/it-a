# Unit Test for AnomalyClusteringFactorExtractor_Vppmicrogrids (Virtual Power Plants & Microgrid Orchestration).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.vpp_microgrids.anomaly_clustering_factor import AnomalyClusteringFactorExtractor_Vppmicrogrids
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_anomaly_clustering_factor_vpp_microgrids_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AnomalyClusteringFactorExtractor_Vppmicrogrids()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"anomaly_clustering_factor_vpp_microgrids_signal" in res.columns
    assert f"anomaly_clustering_factor_vpp_microgrids_risk_score" in res.columns
    assert not res[f"anomaly_clustering_factor_vpp_microgrids_signal"].isnull().any()

def test_anomaly_clustering_factor_vpp_microgrids_empty():
    extractor = AnomalyClusteringFactorExtractor_Vppmicrogrids()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
