# Unit Test for BeamScintillationIndexExtractor_Precisionagriculturehyperspectral (Drone SWIR Crop Water Stress Imaging).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.precision_agriculture_hyperspectral.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Precisionagriculturehyperspectral
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_precision_agriculture_hyperspectral_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Precisionagriculturehyperspectral()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_precision_agriculture_hyperspectral_signal" in res.columns
    assert f"optical_beam_scintillation_index_precision_agriculture_hyperspectral_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_precision_agriculture_hyperspectral_signal"].isnull().any()

def test_optical_beam_scintillation_index_precision_agriculture_hyperspectral_empty():
    extractor = BeamScintillationIndexExtractor_Precisionagriculturehyperspectral()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
