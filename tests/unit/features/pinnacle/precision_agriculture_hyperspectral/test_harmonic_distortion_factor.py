# Unit Test for HarmonicDistortionFactorExtractor_Precisionagriculturehyperspectral (Drone SWIR Crop Water Stress Imaging).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.precision_agriculture_hyperspectral.harmonic_distortion_factor import HarmonicDistortionFactorExtractor_Precisionagriculturehyperspectral
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harmonic_distortion_factor_precision_agriculture_hyperspectral_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarmonicDistortionFactorExtractor_Precisionagriculturehyperspectral()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harmonic_distortion_factor_precision_agriculture_hyperspectral_signal" in res.columns
    assert f"harmonic_distortion_factor_precision_agriculture_hyperspectral_risk_score" in res.columns
    assert not res[f"harmonic_distortion_factor_precision_agriculture_hyperspectral_signal"].isnull().any()

def test_harmonic_distortion_factor_precision_agriculture_hyperspectral_empty():
    extractor = HarmonicDistortionFactorExtractor_Precisionagriculturehyperspectral()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
