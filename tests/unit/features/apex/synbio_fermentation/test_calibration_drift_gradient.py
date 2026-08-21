# Unit Test for CalibrationDriftGradientExtractor_Synbiofermentation (Synthetic Biology & Bioreactor Fermentation).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.synbio_fermentation.calibration_drift_gradient import CalibrationDriftGradientExtractor_Synbiofermentation
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_synbio_fermentation_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Synbiofermentation()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_synbio_fermentation_signal" in res.columns
    assert f"calibration_drift_gradient_synbio_fermentation_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_synbio_fermentation_signal"].isnull().any()

def test_calibration_drift_gradient_synbio_fermentation_empty():
    extractor = CalibrationDriftGradientExtractor_Synbiofermentation()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
