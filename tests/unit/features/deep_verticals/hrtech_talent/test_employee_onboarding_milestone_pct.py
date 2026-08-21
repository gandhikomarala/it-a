# Comprehensive Unit Test for OnboardingMilestoneExtractor (HRTech & People Analytics SaaS).
import pytest
import numpy as np
import pandas as pd
from ml.features.deep_verticals.hrtech_talent.employee_onboarding_milestone_pct import OnboardingMilestoneExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_employee_onboarding_milestone_pct_pipeline():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(35)
    
    extractor = OnboardingMilestoneExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 35
    assert f"employee_onboarding_milestone_pct_signal" in res.columns
    assert f"employee_onboarding_milestone_pct_risk_score" in res.columns
    assert not res[f"employee_onboarding_milestone_pct_signal"].isnull().any()

def test_employee_onboarding_milestone_pct_empty():
    extractor = OnboardingMilestoneExtractor()
    df_empty = pd.DataFrame()
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
