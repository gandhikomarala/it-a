# Unit Test for TelemetryStabilityIndexExtractor_Droneswarmmeshnetworks (Autonomous Drone Swarm Ad-Hoc Mesh).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.drone_swarm_mesh_networks.telemetry_stability_index import TelemetryStabilityIndexExtractor_Droneswarmmeshnetworks
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_drone_swarm_mesh_networks_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Droneswarmmeshnetworks()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_drone_swarm_mesh_networks_signal" in res.columns
    assert f"telemetry_stability_index_drone_swarm_mesh_networks_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_drone_swarm_mesh_networks_signal"].isnull().any()

def test_telemetry_stability_index_drone_swarm_mesh_networks_empty():
    extractor = TelemetryStabilityIndexExtractor_Droneswarmmeshnetworks()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
