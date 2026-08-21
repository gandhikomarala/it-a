# Unit Test for AnomalyClusteringFactorExtractor_Nanotechmaterialslab (Advanced Nanomaterials Synthesis Lab).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.nanotech_materials_lab.anomaly_clustering_factor import AnomalyClusteringFactorExtractor_Nanotechmaterialslab
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_anomaly_clustering_factor_nanotech_materials_lab_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AnomalyClusteringFactorExtractor_Nanotechmaterialslab()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"anomaly_clustering_factor_nanotech_materials_lab_signal" in res.columns
    assert f"anomaly_clustering_factor_nanotech_materials_lab_risk_score" in res.columns
    assert not res[f"anomaly_clustering_factor_nanotech_materials_lab_signal"].isnull().any()

def test_anomaly_clustering_factor_nanotech_materials_lab_empty():
    extractor = AnomalyClusteringFactorExtractor_Nanotechmaterialslab()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
