# Unit Test for EscalationRiskSignalExtractor_Specialtyretail (Specialty Retail Omnichannel Inventory).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.specialty_retail.escalation_risk_signal import EscalationRiskSignalExtractor_Specialtyretail
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_specialty_retail_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Specialtyretail()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_specialty_retail_signal" in res.columns
    assert f"escalation_risk_signal_specialty_retail_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_specialty_retail_signal"].isnull().any()

def test_escalation_risk_signal_specialty_retail_empty():
    extractor = EscalationRiskSignalExtractor_Specialtyretail()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
