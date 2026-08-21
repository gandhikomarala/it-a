# Unit Test for SatisfactionDriftDeltaExtractor_Investmentbanking (Investment Banking M&A Deal Pipeline).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.investment_banking.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Investmentbanking
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_investment_banking_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Investmentbanking()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_investment_banking_signal" in res.columns
    assert f"satisfaction_drift_delta_investment_banking_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_investment_banking_signal"].isnull().any()

def test_satisfaction_drift_delta_investment_banking_empty():
    extractor = SatisfactionDriftDeltaExtractor_Investmentbanking()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
