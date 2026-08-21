# Unit test for APIThroughputDecayExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.api_throughput_decay import APIThroughputDecayExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_api_throughput_decay_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = APIThroughputDecayExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
