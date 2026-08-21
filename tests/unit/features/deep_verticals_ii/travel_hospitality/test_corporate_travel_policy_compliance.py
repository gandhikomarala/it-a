# Comprehensive Unit Test for CorporateTravelComplianceExtractor (Travel, Airline & Hospitality).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals_ii.travel_hospitality.corporate_travel_policy_compliance import CorporateTravelComplianceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_corporate_travel_policy_compliance_pipeline_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CorporateTravelComplianceExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"corporate_travel_policy_compliance_signal" in res.columns
    assert f"corporate_travel_policy_compliance_risk_score" in res.columns
    assert not res[f"corporate_travel_policy_compliance_signal"].isnull().any()

def test_corporate_travel_policy_compliance_empty_handling():
    extractor = CorporateTravelComplianceExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
