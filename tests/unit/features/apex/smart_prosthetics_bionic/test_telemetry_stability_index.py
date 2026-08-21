# Unit Test for TelemetryStabilityIndexExtractor_Smartprostheticsbionic (Myoelectric Bionic Prosthetics Control).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.smart_prosthetics_bionic.telemetry_stability_index import TelemetryStabilityIndexExtractor_Smartprostheticsbionic
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_smart_prosthetics_bionic_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Smartprostheticsbionic()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_smart_prosthetics_bionic_signal" in res.columns
    assert f"telemetry_stability_index_smart_prosthetics_bionic_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_smart_prosthetics_bionic_signal"].isnull().any()

def test_telemetry_stability_index_smart_prosthetics_bionic_empty():
    extractor = TelemetryStabilityIndexExtractor_Smartprostheticsbionic()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
