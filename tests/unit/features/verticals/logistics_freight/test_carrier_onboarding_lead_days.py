# Unit Test for CarrierOnboardingLeadDaysExtractor (Logistics & Supply Chain SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.verticals.logistics_freight.carrier_onboarding_lead_days import CarrierOnboardingLeadDaysExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_carrier_onboarding_lead_days_lifecycle():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = CarrierOnboardingLeadDaysExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"carrier_onboarding_lead_days_signal" in res.columns
    assert f"carrier_onboarding_lead_days_risk_score" in res.columns
    assert not res[f"carrier_onboarding_lead_days_signal"].isnull().any()

def test_carrier_onboarding_lead_days_empty_dataframe():
    extractor = CarrierOnboardingLeadDaysExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
