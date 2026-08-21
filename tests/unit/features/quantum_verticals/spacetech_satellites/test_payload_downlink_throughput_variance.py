# Comprehensive Unit Test for PayloadDownlinkVarianceExtractor (SpaceTech & LEO Satellite Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.quantum_verticals.spacetech_satellites.payload_downlink_throughput_variance import PayloadDownlinkVarianceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_payload_downlink_throughput_variance_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = PayloadDownlinkVarianceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"payload_downlink_throughput_variance_signal" in res.columns
    assert f"payload_downlink_throughput_variance_risk_score" in res.columns
    assert not res[f"payload_downlink_throughput_variance_signal"].isnull().any()

def test_payload_downlink_throughput_variance_empty_handling():
    extractor = PayloadDownlinkVarianceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
