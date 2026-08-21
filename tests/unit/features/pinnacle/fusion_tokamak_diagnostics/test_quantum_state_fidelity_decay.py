# Unit Test for StateFidelityDecayExtractor_Fusiontokamakdiagnostics (Tokamak Fusion Energy Diagnostics).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.fusion_tokamak_diagnostics.quantum_state_fidelity_decay import StateFidelityDecayExtractor_Fusiontokamakdiagnostics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quantum_state_fidelity_decay_fusion_tokamak_diagnostics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = StateFidelityDecayExtractor_Fusiontokamakdiagnostics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"quantum_state_fidelity_decay_fusion_tokamak_diagnostics_signal" in res.columns
    assert f"quantum_state_fidelity_decay_fusion_tokamak_diagnostics_risk_score" in res.columns
    assert not res[f"quantum_state_fidelity_decay_fusion_tokamak_diagnostics_signal"].isnull().any()

def test_quantum_state_fidelity_decay_fusion_tokamak_diagnostics_empty():
    extractor = StateFidelityDecayExtractor_Fusiontokamakdiagnostics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
