# Unit Test for RetentionHealthIndexExtractor_Hydroelectricdamiot (Hydroelectric Dam Structural Health).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.hydroelectric_dam_iot.retention_health_index import RetentionHealthIndexExtractor_Hydroelectricdamiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_hydroelectric_dam_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Hydroelectricdamiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_hydroelectric_dam_iot_signal" in res.columns
    assert f"retention_health_index_hydroelectric_dam_iot_risk_score" in res.columns
    assert not res[f"retention_health_index_hydroelectric_dam_iot_signal"].isnull().any()

def test_retention_health_index_hydroelectric_dam_iot_empty():
    extractor = RetentionHealthIndexExtractor_Hydroelectricdamiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
