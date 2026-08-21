# Unit Test for EISNyquistSlopeExtractor_Cryogenicsuperconductingcavity (Superconducting Particle Accelerator RF).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.cryogenic_superconducting_cavity.electrochemical_impedance_spectroscopy import EISNyquistSlopeExtractor_Cryogenicsuperconductingcavity
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electrochemical_impedance_spectroscopy_cryogenic_superconducting_cavity_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EISNyquistSlopeExtractor_Cryogenicsuperconductingcavity()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electrochemical_impedance_spectroscopy_cryogenic_superconducting_cavity_signal" in res.columns
    assert f"electrochemical_impedance_spectroscopy_cryogenic_superconducting_cavity_risk_score" in res.columns
    assert not res[f"electrochemical_impedance_spectroscopy_cryogenic_superconducting_cavity_signal"].isnull().any()

def test_electrochemical_impedance_spectroscopy_cryogenic_superconducting_cavity_empty():
    extractor = EISNyquistSlopeExtractor_Cryogenicsuperconductingcavity()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
