# Comprehensive Unit Test for ServiceOverdueDaysExtractor (Automotive & Connected Fleet).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.automotive_telematics.service_maintenance_overdue_days import ServiceOverdueDaysExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_service_maintenance_overdue_days_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ServiceOverdueDaysExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"service_maintenance_overdue_days_signal" in res.columns
    assert f"service_maintenance_overdue_days_risk_score" in res.columns
    assert not res[f"service_maintenance_overdue_days_signal"].isnull().any()

def test_service_maintenance_overdue_days_empty_handling():
    extractor = ServiceOverdueDaysExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
