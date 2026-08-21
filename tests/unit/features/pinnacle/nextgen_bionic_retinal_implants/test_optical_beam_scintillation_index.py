# Unit Test for BeamScintillationIndexExtractor_Nextgenbionicretinalimplants (Subretinal Photovoltaic Neural Prosthetics).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.nextgen_bionic_retinal_implants.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Nextgenbionicretinalimplants
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_nextgen_bionic_retinal_implants_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Nextgenbionicretinalimplants()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_nextgen_bionic_retinal_implants_signal" in res.columns
    assert f"optical_beam_scintillation_index_nextgen_bionic_retinal_implants_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_nextgen_bionic_retinal_implants_signal"].isnull().any()

def test_optical_beam_scintillation_index_nextgen_bionic_retinal_implants_empty():
    extractor = BeamScintillationIndexExtractor_Nextgenbionicretinalimplants()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
