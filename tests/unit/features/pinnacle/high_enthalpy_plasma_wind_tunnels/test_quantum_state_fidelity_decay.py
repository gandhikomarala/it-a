# Unit Test for StateFidelityDecayExtractor_Highenthalpyplasmawindtunnels (Arc-Jet Reentry Thermal Protection Testing).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.high_enthalpy_plasma_wind_tunnels.quantum_state_fidelity_decay import StateFidelityDecayExtractor_Highenthalpyplasmawindtunnels
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quantum_state_fidelity_decay_high_enthalpy_plasma_wind_tunnels_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = StateFidelityDecayExtractor_Highenthalpyplasmawindtunnels()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"quantum_state_fidelity_decay_high_enthalpy_plasma_wind_tunnels_signal" in res.columns
    assert f"quantum_state_fidelity_decay_high_enthalpy_plasma_wind_tunnels_risk_score" in res.columns
    assert not res[f"quantum_state_fidelity_decay_high_enthalpy_plasma_wind_tunnels_signal"].isnull().any()

def test_quantum_state_fidelity_decay_high_enthalpy_plasma_wind_tunnels_empty():
    extractor = StateFidelityDecayExtractor_Highenthalpyplasmawindtunnels()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
