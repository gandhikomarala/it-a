# Unit Test for FeatureAdoptionBreadthExtractor_Airlinerevenuemgmt (Airline Yield & Revenue Management).
import pytest
import numpy as np
import pandas as pd
from ml.features.omni.airline_revenue_mgmt.feature_adoption_breadth import FeatureAdoptionBreadthExtractor_Airlinerevenuemgmt
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_adoption_breadth_airline_revenue_mgmt_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = FeatureAdoptionBreadthExtractor_Airlinerevenuemgmt()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"feature_adoption_breadth_airline_revenue_mgmt_signal" in res.columns
    assert f"feature_adoption_breadth_airline_revenue_mgmt_risk_score" in res.columns
    assert not res[f"feature_adoption_breadth_airline_revenue_mgmt_signal"].isnull().any()

def test_feature_adoption_breadth_airline_revenue_mgmt_empty():
    extractor = FeatureAdoptionBreadthExtractor_Airlinerevenuemgmt()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
