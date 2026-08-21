# Unit Test for BeamScintillationIndexExtractor_Highenthalpyplasmawindtunnels (Arc-Jet Reentry Thermal Protection Testing).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.high_enthalpy_plasma_wind_tunnels.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Highenthalpyplasmawindtunnels
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_high_enthalpy_plasma_wind_tunnels_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Highenthalpyplasmawindtunnels()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_high_enthalpy_plasma_wind_tunnels_signal" in res.columns
    assert f"optical_beam_scintillation_index_high_enthalpy_plasma_wind_tunnels_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_high_enthalpy_plasma_wind_tunnels_signal"].isnull().any()

def test_optical_beam_scintillation_index_high_enthalpy_plasma_wind_tunnels_empty():
    extractor = BeamScintillationIndexExtractor_Highenthalpyplasmawindtunnels()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
