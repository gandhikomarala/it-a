# Unit Test for CalibrationDriftGradientExtractor_Verticalfarmingiot (Controlled Environment Vertical Agriculture).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.vertical_farming_iot.calibration_drift_gradient import CalibrationDriftGradientExtractor_Verticalfarmingiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_vertical_farming_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Verticalfarmingiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_vertical_farming_iot_signal" in res.columns
    assert f"calibration_drift_gradient_vertical_farming_iot_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_vertical_farming_iot_signal"].isnull().any()

def test_calibration_drift_gradient_vertical_farming_iot_empty():
    extractor = CalibrationDriftGradientExtractor_Verticalfarmingiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
