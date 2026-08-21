# Unit Test for CriticalToleranceBreachExtractor_Medicalimplantsiot (Connected Medical Implants & Bio-Telemetry).
import pytest
import numpy as np
import pandas as pd
from ml.features.apex.medical_implants_iot.critical_tolerance_breach import CriticalToleranceBreachExtractor_Medicalimplantsiot
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_critical_tolerance_breach_medical_implants_iot_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CriticalToleranceBreachExtractor_Medicalimplantsiot()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"critical_tolerance_breach_medical_implants_iot_signal" in res.columns
    assert f"critical_tolerance_breach_medical_implants_iot_risk_score" in res.columns
    assert not res[f"critical_tolerance_breach_medical_implants_iot_signal"].isnull().any()

def test_critical_tolerance_breach_medical_implants_iot_empty():
    extractor = CriticalToleranceBreachExtractor_Medicalimplantsiot()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
