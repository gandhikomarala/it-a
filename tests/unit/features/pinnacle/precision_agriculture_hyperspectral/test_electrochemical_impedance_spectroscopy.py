# Unit Test for EISNyquistSlopeExtractor_Precisionagriculturehyperspectral (Drone SWIR Crop Water Stress Imaging).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.precision_agriculture_hyperspectral.electrochemical_impedance_spectroscopy import EISNyquistSlopeExtractor_Precisionagriculturehyperspectral
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electrochemical_impedance_spectroscopy_precision_agriculture_hyperspectral_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EISNyquistSlopeExtractor_Precisionagriculturehyperspectral()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electrochemical_impedance_spectroscopy_precision_agriculture_hyperspectral_signal" in res.columns
    assert f"electrochemical_impedance_spectroscopy_precision_agriculture_hyperspectral_risk_score" in res.columns
    assert not res[f"electrochemical_impedance_spectroscopy_precision_agriculture_hyperspectral_signal"].isnull().any()

def test_electrochemical_impedance_spectroscopy_precision_agriculture_hyperspectral_empty():
    extractor = EISNyquistSlopeExtractor_Precisionagriculturehyperspectral()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
