# Unit Test for EscalationRiskSignalExtractor_Investmentbanking (Investment Banking M&A Deal Pipeline).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.investment_banking.escalation_risk_signal import EscalationRiskSignalExtractor_Investmentbanking
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_investment_banking_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Investmentbanking()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_investment_banking_signal" in res.columns
    assert f"escalation_risk_signal_investment_banking_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_investment_banking_signal"].isnull().any()

def test_escalation_risk_signal_investment_banking_empty():
    extractor = EscalationRiskSignalExtractor_Investmentbanking()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
