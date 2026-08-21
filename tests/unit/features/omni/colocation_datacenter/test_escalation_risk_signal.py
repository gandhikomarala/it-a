# Unit Test for EscalationRiskSignalExtractor_Colocationdatacenter (Wholesale Colocation Datacenter Power).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.colocation_datacenter.escalation_risk_signal import EscalationRiskSignalExtractor_Colocationdatacenter
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_colocation_datacenter_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Colocationdatacenter()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_colocation_datacenter_signal" in res.columns
    assert f"escalation_risk_signal_colocation_datacenter_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_colocation_datacenter_signal"].isnull().any()

def test_escalation_risk_signal_colocation_datacenter_empty():
    extractor = EscalationRiskSignalExtractor_Colocationdatacenter()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
