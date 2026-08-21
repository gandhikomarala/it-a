# Unit Test for CalibrationDriftGradientExtractor_Spaceflightlaunch (Commercial Space Flight Launch Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.spaceflight_launch.calibration_drift_gradient import CalibrationDriftGradientExtractor_Spaceflightlaunch
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_spaceflight_launch_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Spaceflightlaunch()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_spaceflight_launch_signal" in res.columns
    assert f"calibration_drift_gradient_spaceflight_launch_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_spaceflight_launch_signal"].isnull().any()

def test_calibration_drift_gradient_spaceflight_launch_empty():
    extractor = CalibrationDriftGradientExtractor_Spaceflightlaunch()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
