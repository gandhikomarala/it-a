# Unit Test for EscalationRiskSignalExtractor_Optometryvision (Optometry & Optical Retail Chain).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.optometry_vision.escalation_risk_signal import EscalationRiskSignalExtractor_Optometryvision
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_optometry_vision_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Optometryvision()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_optometry_vision_signal" in res.columns
    assert f"escalation_risk_signal_optometry_vision_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_optometry_vision_signal"].isnull().any()

def test_escalation_risk_signal_optometry_vision_empty():
    extractor = EscalationRiskSignalExtractor_Optometryvision()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
