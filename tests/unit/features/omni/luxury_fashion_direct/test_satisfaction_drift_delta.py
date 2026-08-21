# Unit Test for SatisfactionDriftDeltaExtractor_Luxuryfashiondirect (Luxury Fashion Direct-to-Consumer).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.luxury_fashion_direct.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Luxuryfashiondirect
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_luxury_fashion_direct_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Luxuryfashiondirect()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_luxury_fashion_direct_signal" in res.columns
    assert f"satisfaction_drift_delta_luxury_fashion_direct_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_luxury_fashion_direct_signal"].isnull().any()

def test_satisfaction_drift_delta_luxury_fashion_direct_empty():
    extractor = SatisfactionDriftDeltaExtractor_Luxuryfashiondirect()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
