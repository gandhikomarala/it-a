# Unit test for FeatureBreadthExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.feature_breadth import FeatureBreadthExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_feature_breadth_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = FeatureBreadthExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
