# Unit Test for BeamScintillationIndexExtractor_Commercialevtolurbanair (All-Electric eVTOL Urban Air Mobility).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.commercial_evtol_urban_air.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Commercialevtolurbanair
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_commercial_evtol_urban_air_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Commercialevtolurbanair()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_commercial_evtol_urban_air_signal" in res.columns
    assert f"optical_beam_scintillation_index_commercial_evtol_urban_air_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_commercial_evtol_urban_air_signal"].isnull().any()

def test_optical_beam_scintillation_index_commercial_evtol_urban_air_empty():
    extractor = BeamScintillationIndexExtractor_Commercialevtolurbanair()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
