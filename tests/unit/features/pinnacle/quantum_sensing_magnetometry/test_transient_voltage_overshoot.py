# Unit Test for TransientVoltageOvershootExtractor_Quantumsensingmagnetometry (Nitrogen-Vacancy Quantum Magnetometry).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.quantum_sensing_magnetometry.transient_voltage_overshoot import TransientVoltageOvershootExtractor_Quantumsensingmagnetometry
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_transient_voltage_overshoot_quantum_sensing_magnetometry_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TransientVoltageOvershootExtractor_Quantumsensingmagnetometry()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"transient_voltage_overshoot_quantum_sensing_magnetometry_signal" in res.columns
    assert f"transient_voltage_overshoot_quantum_sensing_magnetometry_risk_score" in res.columns
    assert not res[f"transient_voltage_overshoot_quantum_sensing_magnetometry_signal"].isnull().any()

def test_transient_voltage_overshoot_quantum_sensing_magnetometry_empty():
    extractor = TransientVoltageOvershootExtractor_Quantumsensingmagnetometry()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
