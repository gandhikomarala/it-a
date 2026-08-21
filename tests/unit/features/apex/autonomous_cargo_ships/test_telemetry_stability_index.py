# Unit Test for TelemetryStabilityIndexExtractor_Autonomouscargoships (Autonomous Trans-Oceanic Cargo Ships).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.autonomous_cargo_ships.telemetry_stability_index import TelemetryStabilityIndexExtractor_Autonomouscargoships
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_autonomous_cargo_ships_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Autonomouscargoships()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_autonomous_cargo_ships_signal" in res.columns
    assert f"telemetry_stability_index_autonomous_cargo_ships_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_autonomous_cargo_ships_signal"].isnull().any()

def test_telemetry_stability_index_autonomous_cargo_ships_empty():
    extractor = TelemetryStabilityIndexExtractor_Autonomouscargoships()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
