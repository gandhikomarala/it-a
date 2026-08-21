# Comprehensive Unit Test for OTASoftwareUpdateLagDaysExtractor (Automotive & Connected Fleet).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.automotive_telematics.ota_software_update_lag_days import OTASoftwareUpdateLagDaysExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_ota_software_update_lag_days_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OTASoftwareUpdateLagDaysExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"ota_software_update_lag_days_signal" in res.columns
    assert f"ota_software_update_lag_days_risk_score" in res.columns
    assert not res[f"ota_software_update_lag_days_signal"].isnull().any()

def test_ota_software_update_lag_days_empty_handling():
    extractor = OTASoftwareUpdateLagDaysExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
