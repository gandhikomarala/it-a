# Unit Test for EscalationRiskSignalExtractor_Higheredadmissions (University Admissions & Enrollment).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.higher_ed_admissions.escalation_risk_signal import EscalationRiskSignalExtractor_Higheredadmissions
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_higher_ed_admissions_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Higheredadmissions()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_higher_ed_admissions_signal" in res.columns
    assert f"escalation_risk_signal_higher_ed_admissions_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_higher_ed_admissions_signal"].isnull().any()

def test_escalation_risk_signal_higher_ed_admissions_empty():
    extractor = EscalationRiskSignalExtractor_Higheredadmissions()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
