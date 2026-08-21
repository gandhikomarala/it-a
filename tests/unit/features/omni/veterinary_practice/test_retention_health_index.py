# Unit Test for RetentionHealthIndexExtractor_Veterinarypractice (Veterinary Practice Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.veterinary_practice.retention_health_index import RetentionHealthIndexExtractor_Veterinarypractice
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_veterinary_practice_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Veterinarypractice()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_veterinary_practice_signal" in res.columns
    assert f"retention_health_index_veterinary_practice_risk_score" in res.columns
    assert not res[f"retention_health_index_veterinary_practice_signal"].isnull().any()

def test_retention_health_index_veterinary_practice_empty():
    extractor = RetentionHealthIndexExtractor_Veterinarypractice()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
