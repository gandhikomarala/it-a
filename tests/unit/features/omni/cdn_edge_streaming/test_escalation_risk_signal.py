# Unit Test for EscalationRiskSignalExtractor_Cdnedgestreaming (Global CDN Video Edge Caching).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.cdn_edge_streaming.escalation_risk_signal import EscalationRiskSignalExtractor_Cdnedgestreaming
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_escalation_risk_signal_cdn_edge_streaming_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EscalationRiskSignalExtractor_Cdnedgestreaming()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"escalation_risk_signal_cdn_edge_streaming_signal" in res.columns
    assert f"escalation_risk_signal_cdn_edge_streaming_risk_score" in res.columns
    assert not res[f"escalation_risk_signal_cdn_edge_streaming_signal"].isnull().any()

def test_escalation_risk_signal_cdn_edge_streaming_empty():
    extractor = EscalationRiskSignalExtractor_Cdnedgestreaming()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
