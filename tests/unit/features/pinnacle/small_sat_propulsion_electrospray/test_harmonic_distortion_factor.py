# Unit Test for HarmonicDistortionFactorExtractor_Smallsatpropulsionelectrospray (Colloid Thruster Electrospray Propulsion).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.small_sat_propulsion_electrospray.harmonic_distortion_factor import HarmonicDistortionFactorExtractor_Smallsatpropulsionelectrospray
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harmonic_distortion_factor_small_sat_propulsion_electrospray_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarmonicDistortionFactorExtractor_Smallsatpropulsionelectrospray()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harmonic_distortion_factor_small_sat_propulsion_electrospray_signal" in res.columns
    assert f"harmonic_distortion_factor_small_sat_propulsion_electrospray_risk_score" in res.columns
    assert not res[f"harmonic_distortion_factor_small_sat_propulsion_electrospray_signal"].isnull().any()

def test_harmonic_distortion_factor_small_sat_propulsion_electrospray_empty():
    extractor = HarmonicDistortionFactorExtractor_Smallsatpropulsionelectrospray()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
