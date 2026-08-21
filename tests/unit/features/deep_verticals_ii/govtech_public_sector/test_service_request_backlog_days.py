# Comprehensive Unit Test for ServiceRequestBacklogDaysExtractor (GovTech & Municipal Services).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.govtech_public_sector.service_request_backlog_days import ServiceRequestBacklogDaysExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_service_request_backlog_days_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = ServiceRequestBacklogDaysExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"service_request_backlog_days_signal" in res.columns
    assert f"service_request_backlog_days_risk_score" in res.columns
    assert not res[f"service_request_backlog_days_signal"].isnull().any()

def test_service_request_backlog_days_empty_handling():
    extractor = ServiceRequestBacklogDaysExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
