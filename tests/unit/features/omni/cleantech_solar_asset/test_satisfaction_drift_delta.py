# Unit Test for SatisfactionDriftDeltaExtractor_Cleantechsolarasset (Utility Solar Asset Performance).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cleantech_solar_asset.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Cleantechsolarasset
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_cleantech_solar_asset_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Cleantechsolarasset()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_cleantech_solar_asset_signal" in res.columns
    assert f"satisfaction_drift_delta_cleantech_solar_asset_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_cleantech_solar_asset_signal"].isnull().any()

def test_satisfaction_drift_delta_cleantech_solar_asset_empty():
    extractor = SatisfactionDriftDeltaExtractor_Cleantechsolarasset()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
