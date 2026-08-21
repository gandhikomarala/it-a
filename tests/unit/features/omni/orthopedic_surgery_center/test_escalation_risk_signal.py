# Unit Test for EscalationRiskSignalExtractor_Orthopedicsurgerycenter (Ambulatory Surgical Center Operations).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.orthopedic_surgery_center.escalation_risk_signal import EscalationRiskSignalExtractor_Orthopedicsurgerycenter
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_orthopedic_surgery_center_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Orthopedicsurgerycenter()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_orthopedic_surgery_center_signal" in res.columns
    assert f"escalation_risk_signal_orthopedic_surgery_center_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_orthopedic_surgery_center_signal"].isnull().any()

def test_escalation_risk_signal_orthopedic_surgery_center_empty():
    extractor = EscalationRiskSignalExtractor_Orthopedicsurgerycenter()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
