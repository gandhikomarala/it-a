# Unit Test for BeamScintillationIndexExtractor_Directlithiumextractionbrine (Geothermal Brine Direct Lithium Extraction).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.direct_lithium_extraction_brine.optical_beam_scintillation_index import BeamScintillationIndexExtractor_Directlithiumextractionbrine
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_optical_beam_scintillation_index_direct_lithium_extraction_brine_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = BeamScintillationIndexExtractor_Directlithiumextractionbrine()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"optical_beam_scintillation_index_direct_lithium_extraction_brine_signal" in res.columns
    assert f"optical_beam_scintillation_index_direct_lithium_extraction_brine_risk_score" in res.columns
    assert not res[f"optical_beam_scintillation_index_direct_lithium_extraction_brine_signal"].isnull().any()

def test_optical_beam_scintillation_index_direct_lithium_extraction_brine_empty():
    extractor = BeamScintillationIndexExtractor_Directlithiumextractionbrine()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
