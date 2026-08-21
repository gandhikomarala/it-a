# Unit Test for BeamScintillationIndexExtractor_Orbitaldebrislaserablation (Pulsed Laser Orbital Debris Remediation).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.orbital_debris_laser_ablation.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Orbitaldebrislaserablation
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_orbital_debris_laser_ablation_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Orbitaldebrislaserablation()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_orbital_debris_laser_ablation_signal" in res.columns
    assert f"optical_beam_scintillation_index_orbital_debris_laser_ablation_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_orbital_debris_laser_ablation_signal"].isnull().any()

def test_optical_beam_scintillation_index_orbital_debris_laser_ablation_empty():
    extractor = BeamScintillationIndexExtractor_Orbitaldebrislaserablation()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
