# Unit Test for AnomalyClusteringFactorExtractor_Celltherapycart (Autologous CAR-T Cell Therapy Manufacturing).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.cell_therapy_car_t.anomaly_clustering_factor import AnomalyClusteringFactorExtractor_Celltherapycart
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_anomaly_clustering_factor_cell_therapy_car_t_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = AnomalyClusteringFactorExtractor_Celltherapycart()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"anomaly_clustering_factor_cell_therapy_car_t_signal" in res.columns
    assert f"anomaly_clustering_factor_cell_therapy_car_t_risk_score" in res.columns
    assert not res[f"anomaly_clustering_factor_cell_therapy_car_t_signal"].isnull().any()

def test_anomaly_clustering_factor_cell_therapy_car_t_empty():
    extractor = AnomalyClusteringFactorExtractor_Celltherapycart()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
