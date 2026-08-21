# Unit Test for CalibrationDriftGradientExtractor_Timbersmartforestry (Autonomous Precision Forestry Harvesters).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.timber_smart_forestry.calibration_drift_gradient import CalibrationDriftGradientExtractor_Timbersmartforestry
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_timber_smart_forestry_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Timbersmartforestry()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_timber_smart_forestry_signal" in res.columns
    assert f"calibration_drift_gradient_timber_smart_forestry_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_timber_smart_forestry_signal"].isnull().any()

def test_calibration_drift_gradient_timber_smart_forestry_empty():
    extractor = CalibrationDriftGradientExtractor_Timbersmartforestry()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
