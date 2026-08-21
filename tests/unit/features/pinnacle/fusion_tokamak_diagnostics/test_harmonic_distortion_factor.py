# Unit Test for HarmonicDistortionFactorExtractor_Fusiontokamakdiagnostics (Tokamak Fusion Energy Diagnostics).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.fusion_tokamak_diagnostics.harmonic_distortion_factor import HarmonicDistortionFactorExtractor_Fusiontokamakdiagnostics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harmonic_distortion_factor_fusion_tokamak_diagnostics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarmonicDistortionFactorExtractor_Fusiontokamakdiagnostics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harmonic_distortion_factor_fusion_tokamak_diagnostics_signal" in res.columns
    assert f"harmonic_distortion_factor_fusion_tokamak_diagnostics_risk_score" in res.columns
    assert not res[f"harmonic_distortion_factor_fusion_tokamak_diagnostics_signal"].isnull().any()

def test_harmonic_distortion_factor_fusion_tokamak_diagnostics_empty():
    extractor = HarmonicDistortionFactorExtractor_Fusiontokamakdiagnostics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
