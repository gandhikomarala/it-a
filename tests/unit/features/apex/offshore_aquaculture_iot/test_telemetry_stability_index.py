# Unit Test for TelemetryStabilityIndexExtractor_Offshoreaquacultureiot (Open-Ocean Smart Aquaculture Cages).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.offshore_aquaculture_iot.telemetry_stability_index import TelemetryStabilityIndexExtractor_Offshoreaquacultureiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_offshore_aquaculture_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Offshoreaquacultureiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_offshore_aquaculture_iot_signal" in res.columns
    assert f"telemetry_stability_index_offshore_aquaculture_iot_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_offshore_aquaculture_iot_signal"].isnull().any()

def test_telemetry_stability_index_offshore_aquaculture_iot_empty():
    extractor = TelemetryStabilityIndexExtractor_Offshoreaquacultureiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
