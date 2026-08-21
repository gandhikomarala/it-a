# Unit Test for UsageIntensityRatioExtractor_Airlinerevenuemgmt (Airline Yield & Revenue Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.airline_revenue_mgmt.usage_intensity_ratio import UsageIntensityRatioExtractor_Airlinerevenuemgmt
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_intensity_ratio_airline_revenue_mgmt_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = UsageIntensityRatioExtractor_Airlinerevenuemgmt()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"usage_intensity_ratio_airline_revenue_mgmt_signal" in res.columns
    assert f"usage_intensity_ratio_airline_revenue_mgmt_risk_score" in res.columns
    assert not res[f"usage_intensity_ratio_airline_revenue_mgmt_signal"].isnull().any()

def test_usage_intensity_ratio_airline_revenue_mgmt_empty():
    extractor = UsageIntensityRatioExtractor_Airlinerevenuemgmt()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
