# Unit Test for TelemetryStabilityIndexExtractor_Geostationarycommsatpayload (GEO High-Throughput Satellite Spot Beams).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.geostationary_commsat_payload.telemetry_stability_index import TelemetryStabilityIndexExtractor_Geostationarycommsatpayload
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_geostationary_commsat_payload_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Geostationarycommsatpayload()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_geostationary_commsat_payload_signal" in res.columns
    assert f"telemetry_stability_index_geostationary_commsat_payload_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_geostationary_commsat_payload_signal"].isnull().any()

def test_telemetry_stability_index_geostationary_commsat_payload_empty():
    extractor = TelemetryStabilityIndexExtractor_Geostationarycommsatpayload()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
