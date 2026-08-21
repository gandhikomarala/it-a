# Unit Test for EscalationRiskSignalExtractor_Telecomftthbroadband (FTTH Fiber Gigabit Broadband Access).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.telecom_ftth_broadband.escalation_risk_signal import EscalationRiskSignalExtractor_Telecomftthbroadband
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_telecom_ftth_broadband_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Telecomftthbroadband()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_telecom_ftth_broadband_signal" in res.columns
    assert f"escalation_risk_signal_telecom_ftth_broadband_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_telecom_ftth_broadband_signal"].isnull().any()

def test_escalation_risk_signal_telecom_ftth_broadband_empty():
    extractor = EscalationRiskSignalExtractor_Telecomftthbroadband()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
