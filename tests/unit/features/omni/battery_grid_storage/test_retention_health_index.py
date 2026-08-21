# Unit Test for RetentionHealthIndexExtractor_Batterygridstorage (BESS Utility Battery Energy Storage).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.battery_grid_storage.retention_health_index import RetentionHealthIndexExtractor_Batterygridstorage
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_battery_grid_storage_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Batterygridstorage()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_battery_grid_storage_signal" in res.columns
    assert f"retention_health_index_battery_grid_storage_risk_score" in res.columns
    assert not res[f"retention_health_index_battery_grid_storage_signal"].isnull().any()

def test_retention_health_index_battery_grid_storage_empty():
    extractor = RetentionHealthIndexExtractor_Batterygridstorage()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
