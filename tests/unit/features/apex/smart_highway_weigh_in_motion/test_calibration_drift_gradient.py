# Unit Test for CalibrationDriftGradientExtractor_Smarthighwayweighinmotion (High-Speed Highway Weigh-in-Motion).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_highway_weigh_in_motion.calibration_drift_gradient import CalibrationDriftGradientExtractor_Smarthighwayweighinmotion
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_smart_highway_weigh_in_motion_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Smarthighwayweighinmotion()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_smart_highway_weigh_in_motion_signal" in res.columns
    assert f"calibration_drift_gradient_smart_highway_weigh_in_motion_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_smart_highway_weigh_in_motion_signal"].isnull().any()

def test_calibration_drift_gradient_smart_highway_weigh_in_motion_empty():
    extractor = CalibrationDriftGradientExtractor_Smarthighwayweighinmotion()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
