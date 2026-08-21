# Unit Test for EscalationRiskSignalExtractor_Veterinarypractice (Veterinary Practice Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.veterinary_practice.escalation_risk_signal import EscalationRiskSignalExtractor_Veterinarypractice
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_veterinary_practice_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Veterinarypractice()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_veterinary_practice_signal" in res.columns
    assert f"escalation_risk_signal_veterinary_practice_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_veterinary_practice_signal"].isnull().any()

def test_escalation_risk_signal_veterinary_practice_empty():
    extractor = EscalationRiskSignalExtractor_Veterinarypractice()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
