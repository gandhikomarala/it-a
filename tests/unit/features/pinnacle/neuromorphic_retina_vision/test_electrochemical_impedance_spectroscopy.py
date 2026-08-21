# Unit Test for EISNyquistSlopeExtractor_Neuromorphicretinavision (Event-Based Neuromorphic Silicon Retina).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.neuromorphic_retina_vision.electrochemical_impedance_spectroscopy import EISNyquistSlopeExtractor_Neuromorphicretinavision
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_electrochemical_impedance_spectroscopy_neuromorphic_retina_vision_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EISNyquistSlopeExtractor_Neuromorphicretinavision()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"electrochemical_impedance_spectroscopy_neuromorphic_retina_vision_signal" in res.columns
    assert f"electrochemical_impedance_spectroscopy_neuromorphic_retina_vision_risk_score" in res.columns
    assert not res[f"electrochemical_impedance_spectroscopy_neuromorphic_retina_vision_signal"].isnull().any()

def test_electrochemical_impedance_spectroscopy_neuromorphic_retina_vision_empty():
    extractor = EISNyquistSlopeExtractor_Neuromorphicretinavision()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
