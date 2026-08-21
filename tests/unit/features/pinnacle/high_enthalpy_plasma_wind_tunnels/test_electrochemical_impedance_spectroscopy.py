# Unit Test for EISNyquistSlopeExtractor_Highenthalpyplasmawindtunnels (Arc-Jet Reentry Thermal Protection Testing).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.high_enthalpy_plasma_wind_tunnels.electrochemical_impedance_spectroscopy import EISNyquistSlopeExtractor_Highenthalpyplasmawindtunnels
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electrochemical_impedance_spectroscopy_high_enthalpy_plasma_wind_tunnels_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EISNyquistSlopeExtractor_Highenthalpyplasmawindtunnels()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electrochemical_impedance_spectroscopy_high_enthalpy_plasma_wind_tunnels_signal" in res.columns
    assert f"electrochemical_impedance_spectroscopy_high_enthalpy_plasma_wind_tunnels_risk_score" in res.columns
    assert not res[f"electrochemical_impedance_spectroscopy_high_enthalpy_plasma_wind_tunnels_signal"].isnull().any()

def test_electrochemical_impedance_spectroscopy_high_enthalpy_plasma_wind_tunnels_empty():
    extractor = EISNyquistSlopeExtractor_Highenthalpyplasmawindtunnels()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
