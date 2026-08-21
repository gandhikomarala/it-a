# Unit Test for CalibrationDriftGradientExtractor_Hypersonicaerodynamics (Hypersonic Scramjet Aerothermal Sensors).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.hypersonic_aerodynamics.calibration_drift_gradient import CalibrationDriftGradientExtractor_Hypersonicaerodynamics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_hypersonic_aerodynamics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Hypersonicaerodynamics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_hypersonic_aerodynamics_signal" in res.columns
    assert f"calibration_drift_gradient_hypersonic_aerodynamics_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_hypersonic_aerodynamics_signal"].isnull().any()

def test_calibration_drift_gradient_hypersonic_aerodynamics_empty():
    extractor = CalibrationDriftGradientExtractor_Hypersonicaerodynamics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
