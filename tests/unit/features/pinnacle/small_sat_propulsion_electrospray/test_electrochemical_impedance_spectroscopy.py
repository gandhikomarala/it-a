# Unit Test for EISNyquistSlopeExtractor_Smallsatpropulsionelectrospray (Colloid Thruster Electrospray Propulsion).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.small_sat_propulsion_electrospray.electrochemical_impedance_spectroscopy import EISNyquistSlopeExtractor_Smallsatpropulsionelectrospray
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electrochemical_impedance_spectroscopy_small_sat_propulsion_electrospray_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EISNyquistSlopeExtractor_Smallsatpropulsionelectrospray()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electrochemical_impedance_spectroscopy_small_sat_propulsion_electrospray_signal" in res.columns
    assert f"electrochemical_impedance_spectroscopy_small_sat_propulsion_electrospray_risk_score" in res.columns
    assert not res[f"electrochemical_impedance_spectroscopy_small_sat_propulsion_electrospray_signal"].isnull().any()

def test_electrochemical_impedance_spectroscopy_small_sat_propulsion_electrospray_empty():
    extractor = EISNyquistSlopeExtractor_Smallsatpropulsionelectrospray()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
