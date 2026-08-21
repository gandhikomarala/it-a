# Unit Test for HarmonicDistortionFactorExtractor_Neuromorphicretinavision (Event-Based Neuromorphic Silicon Retina).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.neuromorphic_retina_vision.harmonic_distortion_factor import HarmonicDistortionFactorExtractor_Neuromorphicretinavision
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harmonic_distortion_factor_neuromorphic_retina_vision_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarmonicDistortionFactorExtractor_Neuromorphicretinavision()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harmonic_distortion_factor_neuromorphic_retina_vision_signal" in res.columns
    assert f"harmonic_distortion_factor_neuromorphic_retina_vision_risk_score" in res.columns
    assert not res[f"harmonic_distortion_factor_neuromorphic_retina_vision_signal"].isnull().any()

def test_harmonic_distortion_factor_neuromorphic_retina_vision_empty():
    extractor = HarmonicDistortionFactorExtractor_Neuromorphicretinavision()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
