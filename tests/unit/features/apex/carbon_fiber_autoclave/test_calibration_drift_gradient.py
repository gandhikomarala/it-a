# Unit Test for CalibrationDriftGradientExtractor_Carbonfiberautoclave (Aerospace Carbon Fiber Composite Autoclaves).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.carbon_fiber_autoclave.calibration_drift_gradient import CalibrationDriftGradientExtractor_Carbonfiberautoclave
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_carbon_fiber_autoclave_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Carbonfiberautoclave()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_carbon_fiber_autoclave_signal" in res.columns
    assert f"calibration_drift_gradient_carbon_fiber_autoclave_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_carbon_fiber_autoclave_signal"].isnull().any()

def test_calibration_drift_gradient_carbon_fiber_autoclave_empty():
    extractor = CalibrationDriftGradientExtractor_Carbonfiberautoclave()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
