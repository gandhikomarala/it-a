# Unit Test for CalibrationDriftGradientExtractor_Batterygigafactoryquality (Lithium-Ion Battery Gigafactory Cell QC).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.battery_gigafactory_quality.calibration_drift_gradient import CalibrationDriftGradientExtractor_Batterygigafactoryquality
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_battery_gigafactory_quality_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Batterygigafactoryquality()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_battery_gigafactory_quality_signal" in res.columns
    assert f"calibration_drift_gradient_battery_gigafactory_quality_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_battery_gigafactory_quality_signal"].isnull().any()

def test_calibration_drift_gradient_battery_gigafactory_quality_empty():
    extractor = CalibrationDriftGradientExtractor_Batterygigafactoryquality()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
