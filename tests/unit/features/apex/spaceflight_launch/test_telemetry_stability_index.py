# Unit Test for TelemetryStabilityIndexExtractor_Spaceflightlaunch (Commercial Space Flight Launch Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.spaceflight_launch.telemetry_stability_index import TelemetryStabilityIndexExtractor_Spaceflightlaunch
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_spaceflight_launch_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Spaceflightlaunch()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_spaceflight_launch_signal" in res.columns
    assert f"telemetry_stability_index_spaceflight_launch_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_spaceflight_launch_signal"].isnull().any()

def test_telemetry_stability_index_spaceflight_launch_empty():
    extractor = TelemetryStabilityIndexExtractor_Spaceflightlaunch()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
