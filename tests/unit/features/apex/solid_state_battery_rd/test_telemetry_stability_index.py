# Unit Test for TelemetryStabilityIndexExtractor_Solidstatebatteryrd (All-Solid-State Battery Anode Electrolyte).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.solid_state_battery_rd.telemetry_stability_index import TelemetryStabilityIndexExtractor_Solidstatebatteryrd
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_solid_state_battery_rd_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Solidstatebatteryrd()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_solid_state_battery_rd_signal" in res.columns
    assert f"telemetry_stability_index_solid_state_battery_rd_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_solid_state_battery_rd_signal"].isnull().any()

def test_telemetry_stability_index_solid_state_battery_rd_empty():
    extractor = TelemetryStabilityIndexExtractor_Solidstatebatteryrd()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
