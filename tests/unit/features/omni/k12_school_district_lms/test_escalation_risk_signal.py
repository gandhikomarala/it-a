# Unit Test for EscalationRiskSignalExtractor_K12Schooldistrictlms (K-12 School District LMS Analytics).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.k12_school_district_lms.escalation_risk_signal import EscalationRiskSignalExtractor_K12Schooldistrictlms
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_k12_school_district_lms_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_K12Schooldistrictlms()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_k12_school_district_lms_signal" in res.columns
    assert f"escalation_risk_signal_k12_school_district_lms_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_k12_school_district_lms_signal"].isnull().any()

def test_escalation_risk_signal_k12_school_district_lms_empty():
    extractor = EscalationRiskSignalExtractor_K12Schooldistrictlms()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
