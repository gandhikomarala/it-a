# Unit Test for SatisfactionDriftDeltaExtractor_Batterygridstorage (BESS Utility Battery Energy Storage).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.battery_grid_storage.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Batterygridstorage
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_battery_grid_storage_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Batterygridstorage()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_battery_grid_storage_signal" in res.columns
    assert f"satisfaction_drift_delta_battery_grid_storage_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_battery_grid_storage_signal"].isnull().any()

def test_satisfaction_drift_delta_battery_grid_storage_empty():
    extractor = SatisfactionDriftDeltaExtractor_Batterygridstorage()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
