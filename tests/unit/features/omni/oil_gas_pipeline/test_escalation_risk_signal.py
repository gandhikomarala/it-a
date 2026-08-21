# Unit Test for EscalationRiskSignalExtractor_Oilgaspipeline (Oil & Gas Pipeline Integrity).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.oil_gas_pipeline.escalation_risk_signal import EscalationRiskSignalExtractor_Oilgaspipeline
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_oil_gas_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Oilgaspipeline()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_oil_gas_pipeline_signal" in res.columns
    assert f"escalation_risk_signal_oil_gas_pipeline_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_oil_gas_pipeline_signal"].isnull().any()

def test_escalation_risk_signal_oil_gas_pipeline_empty():
    extractor = EscalationRiskSignalExtractor_Oilgaspipeline()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
