# Unit Test for TransientVoltageOvershootExtractor_Fusiontokamakdiagnostics (Tokamak Fusion Energy Diagnostics).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.fusion_tokamak_diagnostics.transient_voltage_overshoot import TransientVoltageOvershootExtractor_Fusiontokamakdiagnostics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_transient_voltage_overshoot_fusion_tokamak_diagnostics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TransientVoltageOvershootExtractor_Fusiontokamakdiagnostics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"transient_voltage_overshoot_fusion_tokamak_diagnostics_signal" in res.columns
    assert f"transient_voltage_overshoot_fusion_tokamak_diagnostics_risk_score" in res.columns
    assert not res[f"transient_voltage_overshoot_fusion_tokamak_diagnostics_signal"].isnull().any()

def test_transient_voltage_overshoot_fusion_tokamak_diagnostics_empty():
    extractor = TransientVoltageOvershootExtractor_Fusiontokamakdiagnostics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
