# Unit test for LoginVelocityExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.login_velocity import LoginVelocityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_login_velocity_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = LoginVelocityExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
