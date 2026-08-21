# Unit test for OnboardingMilestoneExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.onboarding_milestone import OnboardingMilestoneExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_onboarding_milestone_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = OnboardingMilestoneExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
