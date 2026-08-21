# Unit Test for HarmonicDistortionFactorExtractor_Autonomousunderwatergliders (Oceanographic Autonomous Underwater Gliders).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.autonomous_underwater_gliders.harmonic_distortion_factor import HarmonicDistortionFactorExtractor_Autonomousunderwatergliders
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harmonic_distortion_factor_autonomous_underwater_gliders_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarmonicDistortionFactorExtractor_Autonomousunderwatergliders()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harmonic_distortion_factor_autonomous_underwater_gliders_signal" in res.columns
    assert f"harmonic_distortion_factor_autonomous_underwater_gliders_risk_score" in res.columns
    assert not res[f"harmonic_distortion_factor_autonomous_underwater_gliders_signal"].isnull().any()

def test_harmonic_distortion_factor_autonomous_underwater_gliders_empty():
    extractor = HarmonicDistortionFactorExtractor_Autonomousunderwatergliders()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
