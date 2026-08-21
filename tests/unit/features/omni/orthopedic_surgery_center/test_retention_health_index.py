# Unit Test for RetentionHealthIndexExtractor_Orthopedicsurgerycenter (Ambulatory Surgical Center Operations).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.orthopedic_surgery_center.retention_health_index import RetentionHealthIndexExtractor_Orthopedicsurgerycenter
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_orthopedic_surgery_center_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Orthopedicsurgerycenter()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_orthopedic_surgery_center_signal" in res.columns
    assert f"retention_health_index_orthopedic_surgery_center_risk_score" in res.columns
    assert not res[f"retention_health_index_orthopedic_surgery_center_signal"].isnull().any()

def test_retention_health_index_orthopedic_surgery_center_empty():
    extractor = RetentionHealthIndexExtractor_Orthopedicsurgerycenter()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
