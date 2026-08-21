# Unit Test for SatisfactionDriftDeltaExtractor_Veterinarypractice (Veterinary Practice Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.veterinary_practice.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Veterinarypractice
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_veterinary_practice_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Veterinarypractice()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_veterinary_practice_signal" in res.columns
    assert f"satisfaction_drift_delta_veterinary_practice_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_veterinary_practice_signal"].isnull().any()

def test_satisfaction_drift_delta_veterinary_practice_empty():
    extractor = SatisfactionDriftDeltaExtractor_Veterinarypractice()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
