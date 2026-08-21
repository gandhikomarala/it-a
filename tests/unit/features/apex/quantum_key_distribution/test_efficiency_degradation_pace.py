# Unit Test for EfficiencyDegradationPaceExtractor_Quantumkeydistribution (Quantum Key Distribution (QKD) Networks).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.quantum_key_distribution.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Quantumkeydistribution
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_quantum_key_distribution_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Quantumkeydistribution()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_quantum_key_distribution_signal" in res.columns
    assert f"efficiency_degradation_pace_quantum_key_distribution_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_quantum_key_distribution_signal"].isnull().any()

def test_efficiency_degradation_pace_quantum_key_distribution_empty():
    extractor = EfficiencyDegradationPaceExtractor_Quantumkeydistribution()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
