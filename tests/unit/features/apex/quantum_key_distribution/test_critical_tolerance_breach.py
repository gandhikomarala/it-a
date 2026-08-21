# Unit Test for CriticalToleranceBreachExtractor_Quantumkeydistribution (Quantum Key Distribution (QKD) Networks).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.quantum_key_distribution.critical_tolerance_breach import CriticalToleranceBreachExtractor_Quantumkeydistribution
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_quantum_key_distribution_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Quantumkeydistribution()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_quantum_key_distribution_signal" in res.columns
    assert f"critical_tolerance_breach_quantum_key_distribution_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_quantum_key_distribution_signal"].isnull().any()

def test_critical_tolerance_breach_quantum_key_distribution_empty():
    extractor = CriticalToleranceBreachExtractor_Quantumkeydistribution()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
