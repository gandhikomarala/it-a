# Unit Test for TelemetryStabilityIndexExtractor_Mininghaultrucks (Autonomous Mining Haul Truck Fleets).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.mining_haul_trucks.telemetry_stability_index import TelemetryStabilityIndexExtractor_Mininghaultrucks
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_mining_haul_trucks_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Mininghaultrucks()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_mining_haul_trucks_signal" in res.columns
    assert f"telemetry_stability_index_mining_haul_trucks_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_mining_haul_trucks_signal"].isnull().any()

def test_telemetry_stability_index_mining_haul_trucks_empty():
    extractor = TelemetryStabilityIndexExtractor_Mininghaultrucks()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
