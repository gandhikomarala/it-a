# Unit Test for EfficiencyDegradationPaceExtractor_Medicalimplantsiot (Connected Medical Implants & Bio-Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.medical_implants_iot.efficiency_degradation_pace import EfficiencyDegradationPaceExtractor_Medicalimplantsiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_efficiency_degradation_pace_medical_implants_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = EfficiencyDegradationPaceExtractor_Medicalimplantsiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"efficiency_degradation_pace_medical_implants_iot_signal" in res.columns
    assert f"efficiency_degradation_pace_medical_implants_iot_risk_score" in res.columns
    assert not res[f"efficiency_degradation_pace_medical_implants_iot_signal"].isnull().any()

def test_efficiency_degradation_pace_medical_implants_iot_empty():
    extractor = EfficiencyDegradationPaceExtractor_Medicalimplantsiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
