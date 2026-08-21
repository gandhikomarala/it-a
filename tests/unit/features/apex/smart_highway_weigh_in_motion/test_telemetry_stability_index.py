# Unit Test for TelemetryStabilityIndexExtractor_Smarthighwayweighinmotion (High-Speed Highway Weigh-in-Motion).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_highway_weigh_in_motion.telemetry_stability_index import TelemetryStabilityIndexExtractor_Smarthighwayweighinmotion
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_smart_highway_weigh_in_motion_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Smarthighwayweighinmotion()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_smart_highway_weigh_in_motion_signal" in res.columns
    assert f"telemetry_stability_index_smart_highway_weigh_in_motion_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_smart_highway_weigh_in_motion_signal"].isnull().any()

def test_telemetry_stability_index_smart_highway_weigh_in_motion_empty():
    extractor = TelemetryStabilityIndexExtractor_Smarthighwayweighinmotion()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
