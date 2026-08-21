# Unit Test for RetentionHealthIndexExtractor_Defenseaerospace (Defense & Aerospace Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.defense_aerospace.retention_health_index import RetentionHealthIndexExtractor_Defenseaerospace
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_defense_aerospace_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Defenseaerospace()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_defense_aerospace_signal" in res.columns
    assert f"retention_health_index_defense_aerospace_risk_score" in res.columns
    assert not res[f"retention_health_index_defense_aerospace_signal"].isnull().any()

def test_retention_health_index_defense_aerospace_empty():
    extractor = RetentionHealthIndexExtractor_Defenseaerospace()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
