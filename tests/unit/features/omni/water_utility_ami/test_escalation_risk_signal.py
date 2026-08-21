# Unit Test for EscalationRiskSignalExtractor_Waterutilityami (Municipal Smart Water AMI Network).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.water_utility_ami.escalation_risk_signal import EscalationRiskSignalExtractor_Waterutilityami
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_water_utility_ami_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Waterutilityami()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_water_utility_ami_signal" in res.columns
    assert f"escalation_risk_signal_water_utility_ami_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_water_utility_ami_signal"].isnull().any()

def test_escalation_risk_signal_water_utility_ami_empty():
    extractor = EscalationRiskSignalExtractor_Waterutilityami()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
