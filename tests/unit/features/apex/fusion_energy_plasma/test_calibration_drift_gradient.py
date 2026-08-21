# Unit Test for CalibrationDriftGradientExtractor_Fusionenergyplasma (Tokamak Fusion Energy Plasma Confinement).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.fusion_energy_plasma.calibration_drift_gradient import CalibrationDriftGradientExtractor_Fusionenergyplasma
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_fusion_energy_plasma_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Fusionenergyplasma()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_fusion_energy_plasma_signal" in res.columns
    assert f"calibration_drift_gradient_fusion_energy_plasma_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_fusion_energy_plasma_signal"].isnull().any()

def test_calibration_drift_gradient_fusion_energy_plasma_empty():
    extractor = CalibrationDriftGradientExtractor_Fusionenergyplasma()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
