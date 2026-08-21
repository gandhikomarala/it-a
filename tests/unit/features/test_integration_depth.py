# Unit test for IntegrationDepthExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.integration_depth import IntegrationDepthExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_integration_depth_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = IntegrationDepthExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
