# Unit Test for BeamScintillationIndexExtractor_Smallsatpropulsionelectrospray (Colloid Thruster Electrospray Propulsion).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.small_sat_propulsion_electrospray.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Smallsatpropulsionelectrospray
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_small_sat_propulsion_electrospray_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Smallsatpropulsionelectrospray()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_small_sat_propulsion_electrospray_signal" in res.columns
    assert f"optical_beam_scintillation_index_small_sat_propulsion_electrospray_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_small_sat_propulsion_electrospray_signal"].isnull().any()

def test_optical_beam_scintillation_index_small_sat_propulsion_electrospray_empty():
    extractor = BeamScintillationIndexExtractor_Smallsatpropulsionelectrospray()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
