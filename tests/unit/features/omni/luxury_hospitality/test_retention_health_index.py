# Unit Test for RetentionHealthIndexExtractor_Luxuryhospitality (Luxury Resort Concierge Guest Experience).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.luxury_hospitality.retention_health_index import RetentionHealthIndexExtractor_Luxuryhospitality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_luxury_hospitality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Luxuryhospitality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_luxury_hospitality_signal" in res.columns
    assert f"retention_health_index_luxury_hospitality_risk_score" in res.columns
    assert not res[f"retention_health_index_luxury_hospitality_signal"].isnull().any()

def test_retention_health_index_luxury_hospitality_empty():
    extractor = RetentionHealthIndexExtractor_Luxuryhospitality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
