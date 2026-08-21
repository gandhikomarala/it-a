# Unit test for BillingAnomalyExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.transformations.billing_analytics import BillingAnomalyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_billing_analytics_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = BillingAnomalyExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
