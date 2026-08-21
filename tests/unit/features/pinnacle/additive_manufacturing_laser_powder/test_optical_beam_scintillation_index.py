# Unit Test for BeamScintillationIndexExtractor_Additivemanufacturinglaserpowder (LPBF Metal Additive Manufacturing).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.additive_manufacturing_laser_powder.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Additivemanufacturinglaserpowder
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_additive_manufacturing_laser_powder_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Additivemanufacturinglaserpowder()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_additive_manufacturing_laser_powder_signal" in res.columns
    assert f"optical_beam_scintillation_index_additive_manufacturing_laser_powder_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_additive_manufacturing_laser_powder_signal"].isnull().any()

def test_optical_beam_scintillation_index_additive_manufacturing_laser_powder_empty():
    extractor = BeamScintillationIndexExtractor_Additivemanufacturinglaserpowder()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
