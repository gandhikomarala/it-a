# Unit Test for SystemEntropyScoreExtractor_Satellitesyntheticaperture (Spaceborne Synthetic Aperture Radar (SAR)).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.satellite_synthetic_aperture.system_entropy_score import SystemEntropyScoreExtractor_Satellitesyntheticaperture
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_system_entropy_score_satellite_synthetic_aperture_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SystemEntropyScoreExtractor_Satellitesyntheticaperture()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"system_entropy_score_satellite_synthetic_aperture_signal" in res.columns
    assert f"system_entropy_score_satellite_synthetic_aperture_risk_score" in res.columns
    assert not res[f"system_entropy_score_satellite_synthetic_aperture_signal"].isnull().any()

def test_system_entropy_score_satellite_synthetic_aperture_empty():
    extractor = SystemEntropyScoreExtractor_Satellitesyntheticaperture()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
