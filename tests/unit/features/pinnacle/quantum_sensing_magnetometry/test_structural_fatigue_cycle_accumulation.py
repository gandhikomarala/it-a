# Unit Test for FatigueCycleAccumulationExtractor_Quantumsensingmagnetometry (Nitrogen-Vacancy Quantum Magnetometry).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.quantum_sensing_magnetometry.structural_fatigue_cycle_accumulation import FatigueCycleAccumulationExtractor_Quantumsensingmagnetometry
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_structural_fatigue_cycle_accumulation_quantum_sensing_magnetometry_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FatigueCycleAccumulationExtractor_Quantumsensingmagnetometry()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"structural_fatigue_cycle_accumulation_quantum_sensing_magnetometry_signal" in res.columns
    assert f"structural_fatigue_cycle_accumulation_quantum_sensing_magnetometry_risk_score" in res.columns
    assert not res[f"structural_fatigue_cycle_accumulation_quantum_sensing_magnetometry_signal"].isnull().any()

def test_structural_fatigue_cycle_accumulation_quantum_sensing_magnetometry_empty():
    extractor = FatigueCycleAccumulationExtractor_Quantumsensingmagnetometry()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
