# Unit Test for HarmonicDistortionFactorExtractor_Commercialevtolurbanair (All-Electric eVTOL Urban Air Mobility).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.commercial_evtol_urban_air.harmonic_distortion_factor import HarmonicDistortionFactorExtractor_Commercialevtolurbanair
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harmonic_distortion_factor_commercial_evtol_urban_air_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarmonicDistortionFactorExtractor_Commercialevtolurbanair()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harmonic_distortion_factor_commercial_evtol_urban_air_signal" in res.columns
    assert f"harmonic_distortion_factor_commercial_evtol_urban_air_risk_score" in res.columns
    assert not res[f"harmonic_distortion_factor_commercial_evtol_urban_air_signal"].isnull().any()

def test_harmonic_distortion_factor_commercial_evtol_urban_air_empty():
    extractor = HarmonicDistortionFactorExtractor_Commercialevtolurbanair()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
