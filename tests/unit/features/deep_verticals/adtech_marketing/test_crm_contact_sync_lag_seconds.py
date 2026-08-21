# Comprehensive Unit Test for CRMSyncLagExtractor (AdTech & Growth Marketing SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.adtech_marketing.crm_contact_sync_lag_seconds import CRMSyncLagExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_crm_contact_sync_lag_seconds_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CRMSyncLagExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"crm_contact_sync_lag_seconds_signal" in res.columns
    assert f"crm_contact_sync_lag_seconds_risk_score" in res.columns
    assert not res[f"crm_contact_sync_lag_seconds_signal"].isnull().any()

def test_crm_contact_sync_lag_seconds_empty():
    extractor = CRMSyncLagExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
