# Comprehensive Unit Test for OnboardingChecklistCompletionExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.onboarding_checklist_completion import OnboardingChecklistCompletionExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_onboarding_checklist_completion_instantiation():
    extractor = OnboardingChecklistCompletionExtractor()
    assert extractor.prefix == "onboarding_checklist_completion"

def test_onboarding_checklist_completion_fit_transform_synthetic():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(40)
    
    extractor = OnboardingChecklistCompletionExtractor()
    extractor.fit(df)
    res = extractor.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 40
    assert not res.isnull().any().any()
    
    # Verify generated features exist
    expected_cols = [c for c in res.columns if c.startswith("onboarding_checklist_completion_")]
    assert len(expected_cols) > 0

def test_onboarding_checklist_completion_transform_empty():
    extractor = OnboardingChecklistCompletionExtractor()
    df_empty = pd.DataFrame(columns=["tenure_months", "monthly_charge"])
    res = extractor.fit_transform(df_empty)
    assert len(res) == 0
