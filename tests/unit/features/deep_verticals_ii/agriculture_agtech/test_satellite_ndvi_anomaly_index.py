# Comprehensive Unit Test for NDVIAnomalyIndexExtractor (Agriculture & Precision Farming).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.agriculture_agtech.satellite_ndvi_anomaly_index import NDVIAnomalyIndexExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satellite_ndvi_anomaly_index_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = NDVIAnomalyIndexExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satellite_ndvi_anomaly_index_signal" in res.columns
    assert f"satellite_ndvi_anomaly_index_risk_score" in res.columns
    assert not res[f"satellite_ndvi_anomaly_index_signal"].isnull().any()

def test_satellite_ndvi_anomaly_index_empty_handling():
    extractor = NDVIAnomalyIndexExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
