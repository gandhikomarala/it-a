# Comprehensive Unit Test for QoSFlowPacketDropVarianceExtractor (Telecom 5G Network Slicing).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.telecom_5g_slicing.qos_flow_packet_drop_variance import QoSFlowPacketDropVarianceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_qos_flow_packet_drop_variance_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = QoSFlowPacketDropVarianceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"qos_flow_packet_drop_variance_signal" in res.columns
    assert f"qos_flow_packet_drop_variance_risk_score" in res.columns
    assert not res[f"qos_flow_packet_drop_variance_signal"].isnull().any()

def test_qos_flow_packet_drop_variance_empty_handling():
    extractor = QoSFlowPacketDropVarianceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
