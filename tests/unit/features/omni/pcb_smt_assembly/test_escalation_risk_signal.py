# Unit Test for EscalationRiskSignalExtractor_Pcbsmtassembly (Surface Mount PCB Assembly Quality).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.pcb_smt_assembly.escalation_risk_signal import EscalationRiskSignalExtractor_Pcbsmtassembly
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_pcb_smt_assembly_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Pcbsmtassembly()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_pcb_smt_assembly_signal" in res.columns
    assert f"escalation_risk_signal_pcb_smt_assembly_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_pcb_smt_assembly_signal"].isnull().any()

def test_escalation_risk_signal_pcb_smt_assembly_empty():
    extractor = EscalationRiskSignalExtractor_Pcbsmtassembly()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
