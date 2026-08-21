# Unit Test for EISNyquistSlopeExtractor_Fusiontokamakdiagnostics (Tokamak Fusion Energy Diagnostics).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.fusion_tokamak_diagnostics.electrochemical_impedance_spectroscopy import EISNyquistSlopeExtractor_Fusiontokamakdiagnostics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electrochemical_impedance_spectroscopy_fusion_tokamak_diagnostics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EISNyquistSlopeExtractor_Fusiontokamakdiagnostics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electrochemical_impedance_spectroscopy_fusion_tokamak_diagnostics_signal" in res.columns
    assert f"electrochemical_impedance_spectroscopy_fusion_tokamak_diagnostics_risk_score" in res.columns
    assert not res[f"electrochemical_impedance_spectroscopy_fusion_tokamak_diagnostics_signal"].isnull().any()

def test_electrochemical_impedance_spectroscopy_fusion_tokamak_diagnostics_empty():
    extractor = EISNyquistSlopeExtractor_Fusiontokamakdiagnostics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
