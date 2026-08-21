# Unit test for ExponentialDecayTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.transformations.decay_functions import ExponentialDecayTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_decay_functions_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = ExponentialDecayTransformer()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
