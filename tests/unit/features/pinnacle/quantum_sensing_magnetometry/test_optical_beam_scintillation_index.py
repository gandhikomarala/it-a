# Unit Test for BeamScintillationIndexExtractor_Quantumsensingmagnetometry (Nitrogen-Vacancy Quantum Magnetometry).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.quantum_sensing_magnetometry.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Quantumsensingmagnetometry
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_quantum_sensing_magnetometry_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Quantumsensingmagnetometry()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_quantum_sensing_magnetometry_signal" in res.columns
    assert f"optical_beam_scintillation_index_quantum_sensing_magnetometry_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_quantum_sensing_magnetometry_signal"].isnull().any()

def test_optical_beam_scintillation_index_quantum_sensing_magnetometry_empty():
    extractor = BeamScintillationIndexExtractor_Quantumsensingmagnetometry()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
