# Unit Test for EISNyquistSlopeExtractor_Orbitaldebrislaserablation (Pulsed Laser Orbital Debris Remediation).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.orbital_debris_laser_ablation.electrochemical_impedance_spectroscopy import EISNyquistSlopeExtractor_Orbitaldebrislaserablation
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electrochemical_impedance_spectroscopy_orbital_debris_laser_ablation_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EISNyquistSlopeExtractor_Orbitaldebrislaserablation()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electrochemical_impedance_spectroscopy_orbital_debris_laser_ablation_signal" in res.columns
    assert f"electrochemical_impedance_spectroscopy_orbital_debris_laser_ablation_risk_score" in res.columns
    assert not res[f"electrochemical_impedance_spectroscopy_orbital_debris_laser_ablation_signal"].isnull().any()

def test_electrochemical_impedance_spectroscopy_orbital_debris_laser_ablation_empty():
    extractor = EISNyquistSlopeExtractor_Orbitaldebrislaserablation()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
