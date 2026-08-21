# Unit Test for StateFidelityDecayExtractor_Directlithiumextractionbrine (Geothermal Brine Direct Lithium Extraction).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.direct_lithium_extraction_brine.quantum_state_fidelity_decay import StateFidelityDecayExtractor_Directlithiumextractionbrine
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quantum_state_fidelity_decay_direct_lithium_extraction_brine_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = StateFidelityDecayExtractor_Directlithiumextractionbrine()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"quantum_state_fidelity_decay_direct_lithium_extraction_brine_signal" in res.columns
    assert f"quantum_state_fidelity_decay_direct_lithium_extraction_brine_risk_score" in res.columns
    assert not res[f"quantum_state_fidelity_decay_direct_lithium_extraction_brine_signal"].isnull().any()

def test_quantum_state_fidelity_decay_direct_lithium_extraction_brine_empty():
    extractor = StateFidelityDecayExtractor_Directlithiumextractionbrine()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
