# Unit Test for CalibrationDriftGradientExtractor_Surgicalrobotics (Precision Robotic-Assisted Surgery).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.surgical_robotics.calibration_drift_gradient import CalibrationDriftGradientExtractor_Surgicalrobotics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_surgical_robotics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Surgicalrobotics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_surgical_robotics_signal" in res.columns
    assert f"calibration_drift_gradient_surgical_robotics_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_surgical_robotics_signal"].isnull().any()

def test_calibration_drift_gradient_surgical_robotics_empty():
    extractor = CalibrationDriftGradientExtractor_Surgicalrobotics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
