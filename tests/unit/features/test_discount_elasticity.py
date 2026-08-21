# Unit test for DiscountElasticityExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.discount_elasticity import DiscountElasticityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_discount_elasticity_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = DiscountElasticityExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
