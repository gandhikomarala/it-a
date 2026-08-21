# Unit Test for StateFidelityDecayExtractor_Additivemanufacturinglaserpowder (LPBF Metal Additive Manufacturing).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.additive_manufacturing_laser_powder.quantum_state_fidelity_decay import StateFidelityDecayExtractor_Additivemanufacturinglaserpowder
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quantum_state_fidelity_decay_additive_manufacturing_laser_powder_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = StateFidelityDecayExtractor_Additivemanufacturinglaserpowder()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"quantum_state_fidelity_decay_additive_manufacturing_laser_powder_signal" in res.columns
    assert f"quantum_state_fidelity_decay_additive_manufacturing_laser_powder_risk_score" in res.columns
    assert not res[f"quantum_state_fidelity_decay_additive_manufacturing_laser_powder_signal"].isnull().any()

def test_quantum_state_fidelity_decay_additive_manufacturing_laser_powder_empty():
    extractor = StateFidelityDecayExtractor_Additivemanufacturinglaserpowder()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
