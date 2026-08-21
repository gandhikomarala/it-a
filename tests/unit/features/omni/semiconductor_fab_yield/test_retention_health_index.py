# Unit Test for RetentionHealthIndexExtractor_Semiconductorfabyield (Semiconductor 3nm Wafer Fab Yield).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.semiconductor_fab_yield.retention_health_index import RetentionHealthIndexExtractor_Semiconductorfabyield
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_semiconductor_fab_yield_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Semiconductorfabyield()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_semiconductor_fab_yield_signal" in res.columns
    assert f"retention_health_index_semiconductor_fab_yield_risk_score" in res.columns
    assert not res[f"retention_health_index_semiconductor_fab_yield_signal"].isnull().any()

def test_retention_health_index_semiconductor_fab_yield_empty():
    extractor = RetentionHealthIndexExtractor_Semiconductorfabyield()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
