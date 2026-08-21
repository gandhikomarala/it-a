# Unit Test for CalibrationDriftGradientExtractor_Bioprintingtissuescaffolds (3D Extrusion Bioprinting Organ Scaffolds).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.bioprinting_tissue_scaffolds.calibration_drift_gradient import CalibrationDriftGradientExtractor_Bioprintingtissuescaffolds
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_calibration_drift_gradient_bioprinting_tissue_scaffolds_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CalibrationDriftGradientExtractor_Bioprintingtissuescaffolds()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"calibration_drift_gradient_bioprinting_tissue_scaffolds_signal" in res.columns
    assert f"calibration_drift_gradient_bioprinting_tissue_scaffolds_risk_score" in res.columns
    assert not res[f"calibration_drift_gradient_bioprinting_tissue_scaffolds_signal"].isnull().any()

def test_calibration_drift_gradient_bioprinting_tissue_scaffolds_empty():
    extractor = CalibrationDriftGradientExtractor_Bioprintingtissuescaffolds()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
