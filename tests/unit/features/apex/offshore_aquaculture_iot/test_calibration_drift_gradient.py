# Unit Test for CalibrationDriftGradientExtractor_Offshoreaquacultureiot (Open-Ocean Smart Aquaculture Cages).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.offshore_aquaculture_iot.calibration_drift_gradient import CalibrationDriftGradientExtractor_Offshoreaquacultureiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_offshore_aquaculture_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Offshoreaquacultureiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_offshore_aquaculture_iot_signal" in res.columns
    assert f"calibration_drift_gradient_offshore_aquaculture_iot_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_offshore_aquaculture_iot_signal"].isnull().any()

def test_calibration_drift_gradient_offshore_aquaculture_iot_empty():
    extractor = CalibrationDriftGradientExtractor_Offshoreaquacultureiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
