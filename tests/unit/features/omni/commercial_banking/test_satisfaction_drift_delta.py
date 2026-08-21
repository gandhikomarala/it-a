# Unit Test for SatisfactionDriftDeltaExtractor_Commercialbanking (Commercial Treasury & Syndicated Lending).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.commercial_banking.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Commercialbanking
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_commercial_banking_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Commercialbanking()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_commercial_banking_signal" in res.columns
    assert f"satisfaction_drift_delta_commercial_banking_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_commercial_banking_signal"].isnull().any()

def test_satisfaction_drift_delta_commercial_banking_empty():
    extractor = SatisfactionDriftDeltaExtractor_Commercialbanking()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
