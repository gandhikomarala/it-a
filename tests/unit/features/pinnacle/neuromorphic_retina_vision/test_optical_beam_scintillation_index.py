# Unit Test for BeamScintillationIndexExtractor_Neuromorphicretinavision (Event-Based Neuromorphic Silicon Retina).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.neuromorphic_retina_vision.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Neuromorphicretinavision
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_neuromorphic_retina_vision_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Neuromorphicretinavision()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_neuromorphic_retina_vision_signal" in res.columns
    assert f"optical_beam_scintillation_index_neuromorphic_retina_vision_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_neuromorphic_retina_vision_signal"].isnull().any()

def test_optical_beam_scintillation_index_neuromorphic_retina_vision_empty():
    extractor = BeamScintillationIndexExtractor_Neuromorphicretinavision()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
