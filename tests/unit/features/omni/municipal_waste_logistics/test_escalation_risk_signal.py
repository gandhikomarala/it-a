# Unit Test for EscalationRiskSignalExtractor_Municipalwastelogistics (Municipal Smart Waste Routing).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.municipal_waste_logistics.escalation_risk_signal import EscalationRiskSignalExtractor_Municipalwastelogistics
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_municipal_waste_logistics_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Municipalwastelogistics()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_municipal_waste_logistics_signal" in res.columns
    assert f"escalation_risk_signal_municipal_waste_logistics_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_municipal_waste_logistics_signal"].isnull().any()

def test_escalation_risk_signal_municipal_waste_logistics_empty():
    extractor = EscalationRiskSignalExtractor_Municipalwastelogistics()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
