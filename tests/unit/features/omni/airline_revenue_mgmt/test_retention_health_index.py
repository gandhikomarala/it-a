# Unit Test for RetentionHealthIndexExtractor_Airlinerevenuemgmt (Airline Yield & Revenue Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.airline_revenue_mgmt.retention_health_index import RetentionHealthIndexExtractor_Airlinerevenuemgmt
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_retention_health_index_airline_revenue_mgmt_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = RetentionHealthIndexExtractor_Airlinerevenuemgmt()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"retention_health_index_airline_revenue_mgmt_signal" in res.columns
    assert f"retention_health_index_airline_revenue_mgmt_risk_score" in res.columns
    assert not res[f"retention_health_index_airline_revenue_mgmt_signal"].isnull().any()

def test_retention_health_index_airline_revenue_mgmt_empty():
    extractor = RetentionHealthIndexExtractor_Airlinerevenuemgmt()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
