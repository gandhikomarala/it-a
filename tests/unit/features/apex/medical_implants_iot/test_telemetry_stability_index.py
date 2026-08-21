# Unit Test for TelemetryStabilityIndexExtractor_Medicalimplantsiot (Connected Medical Implants & Bio-Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.medical_implants_iot.telemetry_stability_index import TelemetryStabilityIndexExtractor_Medicalimplantsiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_telemetry_stability_index_medical_implants_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = TelemetryStabilityIndexExtractor_Medicalimplantsiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"telemetry_stability_index_medical_implants_iot_signal" in res.columns
    assert f"telemetry_stability_index_medical_implants_iot_risk_score" in res.columns
    assert not res[f"telemetry_stability_index_medical_implants_iot_signal"].isnull().any()

def test_telemetry_stability_index_medical_implants_iot_empty():
    extractor = TelemetryStabilityIndexExtractor_Medicalimplantsiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
