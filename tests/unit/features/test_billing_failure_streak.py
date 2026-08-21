# Unit test for BillingFailureStreakExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.billing_failure_streak import BillingFailureStreakExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_billing_failure_streak_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = BillingFailureStreakExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
