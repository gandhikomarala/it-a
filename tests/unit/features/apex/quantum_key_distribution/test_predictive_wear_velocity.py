# Unit Test for PredictiveWearVelocityExtractor_Quantumkeydistribution (Quantum Key Distribution (QKD) Networks).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.quantum_key_distribution.predictive_wear_velocity import PredictiveWearVelocityExtractor_Quantumkeydistribution
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_predictive_wear_velocity_quantum_key_distribution_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PredictiveWearVelocityExtractor_Quantumkeydistribution()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"predictive_wear_velocity_quantum_key_distribution_signal" in res.columns
    assert f"predictive_wear_velocity_quantum_key_distribution_risk_score" in res.columns
    assert not res[f"predictive_wear_velocity_quantum_key_distribution_signal"].isnull().any()

def test_predictive_wear_velocity_quantum_key_distribution_empty():
    extractor = PredictiveWearVelocityExtractor_Quantumkeydistribution()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
