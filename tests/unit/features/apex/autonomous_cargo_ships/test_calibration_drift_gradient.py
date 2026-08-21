# Unit Test for CalibrationDriftGradientExtractor_Autonomouscargoships (Autonomous Trans-Oceanic Cargo Ships).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.autonomous_cargo_ships.calibration_drift_gradient import CalibrationDriftGradientExtractor_Autonomouscargoships
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_autonomous_cargo_ships_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Autonomouscargoships()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_autonomous_cargo_ships_signal" in res.columns
    assert f"calibration_drift_gradient_autonomous_cargo_ships_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_autonomous_cargo_ships_signal"].isnull().any()

def test_calibration_drift_gradient_autonomous_cargo_ships_empty():
    extractor = CalibrationDriftGradientExtractor_Autonomouscargoships()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
