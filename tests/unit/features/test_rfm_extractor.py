# Unit test for RFMFeatureExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.transformations.rfm_extractor import RFMFeatureExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_rfm_extractor_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = RFMFeatureExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
