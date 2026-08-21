# Unit Test for RetentionHealthIndexExtractor_Luxuryfashiondirect (Luxury Fashion Direct-to-Consumer).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.luxury_fashion_direct.retention_health_index import RetentionHealthIndexExtractor_Luxuryfashiondirect
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_luxury_fashion_direct_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Luxuryfashiondirect()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_luxury_fashion_direct_signal" in res.columns
    assert f"retention_health_index_luxury_fashion_direct_risk_score" in res.columns
    assert not res[f"retention_health_index_luxury_fashion_direct_signal"].isnull().any()

def test_retention_health_index_luxury_fashion_direct_empty():
    extractor = RetentionHealthIndexExtractor_Luxuryfashiondirect()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
