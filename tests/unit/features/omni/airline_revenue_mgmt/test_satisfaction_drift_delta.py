# Unit Test for SatisfactionDriftDeltaExtractor_Airlinerevenuemgmt (Airline Yield & Revenue Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.airline_revenue_mgmt.satisfaction_drift_delta import SatisfactionDriftDeltaExtractor_Airlinerevenuemgmt
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_satisfaction_drift_delta_airline_revenue_mgmt_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = SatisfactionDriftDeltaExtractor_Airlinerevenuemgmt()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"satisfaction_drift_delta_airline_revenue_mgmt_signal" in res.columns
    assert f"satisfaction_drift_delta_airline_revenue_mgmt_risk_score" in res.columns
    assert not res[f"satisfaction_drift_delta_airline_revenue_mgmt_signal"].isnull().any()

def test_satisfaction_drift_delta_airline_revenue_mgmt_empty():
    extractor = SatisfactionDriftDeltaExtractor_Airlinerevenuemgmt()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
