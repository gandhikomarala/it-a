# Unit Test for MedicalDeviceIoTConnectivityExtractor (Healthcare & MedTech SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.healthcare_medtech.medical_device_iot_connectivity import MedicalDeviceIoTConnectivityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_medical_device_iot_connectivity_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = MedicalDeviceIoTConnectivityExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"medical_device_iot_connectivity_signal" in res.columns
    assert f"medical_device_iot_connectivity_risk_score" in res.columns
    assert not res[f"medical_device_iot_connectivity_signal"].isnull().any()

def test_medical_device_iot_connectivity_empty_dataframe():
    extractor = MedicalDeviceIoTConnectivityExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
