# Unit test for CrossProductElasticityExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.transformations.cross_product import CrossProductElasticityExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_cross_product_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = CrossProductElasticityExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
