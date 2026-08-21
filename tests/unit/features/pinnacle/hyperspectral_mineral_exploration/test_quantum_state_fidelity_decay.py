# Unit Test for StateFidelityDecayExtractor_Hyperspectralmineralexploration (Airborne Hyperspectral Mineral Mapping).
import pytest
import numpy as np
import pandas as pd
from ml.features.pinnacle.hyperspectral_mineral_exploration.quantum_state_fidelity_decay import StateFidelityDecayExtractor_Hyperspectralmineralexploration
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quantum_state_fidelity_decay_hyperspectral_mineral_exploration_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = StateFidelityDecayExtractor_Hyperspectralmineralexploration()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"quantum_state_fidelity_decay_hyperspectral_mineral_exploration_signal" in res.columns
    assert f"quantum_state_fidelity_decay_hyperspectral_mineral_exploration_risk_score" in res.columns
    assert not res[f"quantum_state_fidelity_decay_hyperspectral_mineral_exploration_signal"].isnull().any()

def test_quantum_state_fidelity_decay_hyperspectral_mineral_exploration_empty():
    extractor = StateFidelityDecayExtractor_Hyperspectralmineralexploration()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
