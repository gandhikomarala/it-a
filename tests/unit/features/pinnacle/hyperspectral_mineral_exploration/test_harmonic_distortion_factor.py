# Unit Test for HarmonicDistortionFactorExtractor_Hyperspectralmineralexploration (Airborne Hyperspectral Mineral Mapping).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.hyperspectral_mineral_exploration.harmonic_distortion_factor import HarmonicDistortionFactorExtractor_Hyperspectralmineralexploration
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_harmonic_distortion_factor_hyperspectral_mineral_exploration_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = HarmonicDistortionFactorExtractor_Hyperspectralmineralexploration()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"harmonic_distortion_factor_hyperspectral_mineral_exploration_signal" in res.columns
    assert f"harmonic_distortion_factor_hyperspectral_mineral_exploration_risk_score" in res.columns
    assert not res[f"harmonic_distortion_factor_hyperspectral_mineral_exploration_signal"].isnull().any()

def test_harmonic_distortion_factor_hyperspectral_mineral_exploration_empty():
    extractor = HarmonicDistortionFactorExtractor_Hyperspectralmineralexploration()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
