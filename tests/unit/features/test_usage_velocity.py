# Unit test for UsageVelocityExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.transformations.usage_velocity import UsageVelocityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_usage_velocity_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = UsageVelocityExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
