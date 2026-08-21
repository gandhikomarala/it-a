# Unit Test for LabResultDeliveryLatencyExtractor (Healthcare & MedTech SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.healthcare_medtech.lab_result_delivery_latency import LabResultDeliveryLatencyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lab_result_delivery_latency_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = LabResultDeliveryLatencyExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"lab_result_delivery_latency_signal" in res.columns
    assert f"lab_result_delivery_latency_risk_score" in res.columns
    assert not res[f"lab_result_delivery_latency_signal"].isnull().any()

def test_lab_result_delivery_latency_empty_dataframe():
    extractor = LabResultDeliveryLatencyExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
