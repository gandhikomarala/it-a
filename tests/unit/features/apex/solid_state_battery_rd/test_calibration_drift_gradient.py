# Unit Test for CalibrationDriftGradientExtractor_Solidstatebatteryrd (All-Solid-State Battery Anode Electrolyte).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.solid_state_battery_rd.calibration_drift_gradient import CalibrationDriftGradientExtractor_Solidstatebatteryrd
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_solid_state_battery_rd_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Solidstatebatteryrd()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_solid_state_battery_rd_signal" in res.columns
    assert f"calibration_drift_gradient_solid_state_battery_rd_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_solid_state_battery_rd_signal"].isnull().any()

def test_calibration_drift_gradient_solid_state_battery_rd_empty():
    extractor = CalibrationDriftGradientExtractor_Solidstatebatteryrd()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
