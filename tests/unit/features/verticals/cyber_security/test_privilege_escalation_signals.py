# Unit Test for PrivilegeEscalationSignalsExtractor (Cyber Security & Threat Intelligence).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.cyber_security.privilege_escalation_signals import PrivilegeEscalationSignalsExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_privilege_escalation_signals_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PrivilegeEscalationSignalsExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"privilege_escalation_signals_signal" in res.columns
    assert f"privilege_escalation_signals_risk_score" in res.columns
    assert not res[f"privilege_escalation_signals_signal"].isnull().any()

def test_privilege_escalation_signals_empty_dataframe():
    extractor = PrivilegeEscalationSignalsExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
