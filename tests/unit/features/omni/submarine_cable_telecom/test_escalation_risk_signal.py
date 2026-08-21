# Unit Test for EscalationRiskSignalExtractor_Submarinecabletelecom (Submarine Fiber Cable Capacity).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.submarine_cable_telecom.escalation_risk_signal import EscalationRiskSignalExtractor_Submarinecabletelecom
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_submarine_cable_telecom_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Submarinecabletelecom()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_submarine_cable_telecom_signal" in res.columns
    assert f"escalation_risk_signal_submarine_cable_telecom_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_submarine_cable_telecom_signal"].isnull().any()

def test_escalation_risk_signal_submarine_cable_telecom_empty():
    extractor = EscalationRiskSignalExtractor_Submarinecabletelecom()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
