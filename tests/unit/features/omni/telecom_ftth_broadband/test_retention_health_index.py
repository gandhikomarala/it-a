# Unit Test for RetentionHealthIndexExtractor_Telecomftthbroadband (FTTH Fiber Gigabit Broadband Access).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.telecom_ftth_broadband.retention_health_index import RetentionHealthIndexExtractor_Telecomftthbroadband
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_telecom_ftth_broadband_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Telecomftthbroadband()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_telecom_ftth_broadband_signal" in res.columns
    assert f"retention_health_index_telecom_ftth_broadband_risk_score" in res.columns
    assert not res[f"retention_health_index_telecom_ftth_broadband_signal"].isnull().any()

def test_retention_health_index_telecom_ftth_broadband_empty():
    extractor = RetentionHealthIndexExtractor_Telecomftthbroadband()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
