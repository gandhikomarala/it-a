# Unit Test for StateFidelityDecayExtractor_Quantumsensingmagnetometry (Nitrogen-Vacancy Quantum Magnetometry).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.quantum_sensing_magnetometry.quantum_state_fidelity_decay import StateFidelityDecayExtractor_Quantumsensingmagnetometry
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quantum_state_fidelity_decay_quantum_sensing_magnetometry_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = StateFidelityDecayExtractor_Quantumsensingmagnetometry()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"quantum_state_fidelity_decay_quantum_sensing_magnetometry_signal" in res.columns
    assert f"quantum_state_fidelity_decay_quantum_sensing_magnetometry_risk_score" in res.columns
    assert not res[f"quantum_state_fidelity_decay_quantum_sensing_magnetometry_signal"].isnull().any()

def test_quantum_state_fidelity_decay_quantum_sensing_magnetometry_empty():
    extractor = StateFidelityDecayExtractor_Quantumsensingmagnetometry()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
