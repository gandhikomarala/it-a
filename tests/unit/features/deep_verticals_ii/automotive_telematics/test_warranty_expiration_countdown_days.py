# Comprehensive Unit Test for WarrantyExpirationDaysExtractor (Automotive & Connected Fleet).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.automotive_telematics.warranty_expiration_countdown_days import WarrantyExpirationDaysExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_warranty_expiration_countdown_days_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = WarrantyExpirationDaysExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"warranty_expiration_countdown_days_signal" in res.columns
    assert f"warranty_expiration_countdown_days_risk_score" in res.columns
    assert not res[f"warranty_expiration_countdown_days_signal"].isnull().any()

def test_warranty_expiration_countdown_days_empty_handling():
    extractor = WarrantyExpirationDaysExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
