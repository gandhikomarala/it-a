# Unit Test for CalibrationDriftGradientExtractor_Semiconductorphotolithography (EUV Semiconductor Photolithography).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.semiconductor_photolithography.calibration_drift_gradient import CalibrationDriftGradientExtractor_Semiconductorphotolithography
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_semiconductor_photolithography_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Semiconductorphotolithography()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_semiconductor_photolithography_signal" in res.columns
    assert f"calibration_drift_gradient_semiconductor_photolithography_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_semiconductor_photolithography_signal"].isnull().any()

def test_calibration_drift_gradient_semiconductor_photolithography_empty():
    extractor = CalibrationDriftGradientExtractor_Semiconductorphotolithography()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
