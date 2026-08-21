# Unit Test for BeamScintillationIndexExtractor_Hyperspectralmineralexploration (Airborne Hyperspectral Mineral Mapping).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.hyperspectral_mineral_exploration.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Hyperspectralmineralexploration
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_hyperspectral_mineral_exploration_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Hyperspectralmineralexploration()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_hyperspectral_mineral_exploration_signal" in res.columns
    assert f"optical_beam_scintillation_index_hyperspectral_mineral_exploration_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_hyperspectral_mineral_exploration_signal"].isnull().any()

def test_optical_beam_scintillation_index_hyperspectral_mineral_exploration_empty():
    extractor = BeamScintillationIndexExtractor_Hyperspectralmineralexploration()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
