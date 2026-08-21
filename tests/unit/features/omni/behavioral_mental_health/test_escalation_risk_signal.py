# Unit Test for EscalationRiskSignalExtractor_Behavioralmentalhealth (Behavioral & Mental Health Telehealth).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.behavioral_mental_health.escalation_risk_signal import EscalationRiskSignalExtractor_Behavioralmentalhealth
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_behavioral_mental_health_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Behavioralmentalhealth()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_behavioral_mental_health_signal" in res.columns
    assert f"escalation_risk_signal_behavioral_mental_health_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_behavioral_mental_health_signal"].isnull().any()

def test_escalation_risk_signal_behavioral_mental_health_empty():
    extractor = EscalationRiskSignalExtractor_Behavioralmentalhealth()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
