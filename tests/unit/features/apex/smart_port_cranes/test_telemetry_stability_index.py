# Unit Test for TelemetryStabilityIndexExtractor_Smartportcranes (Automated Container Port STS Cranes).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_port_cranes.telemetry_stability_index import TelemetryStabilityIndexExtractor_Smartportcranes
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_smart_port_cranes_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Smartportcranes()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_smart_port_cranes_signal" in res.columns
    assert f"telemetry_stability_index_smart_port_cranes_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_smart_port_cranes_signal"].isnull().any()

def test_telemetry_stability_index_smart_port_cranes_empty():
    extractor = TelemetryStabilityIndexExtractor_Smartportcranes()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
