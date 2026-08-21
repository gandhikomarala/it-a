# Unit test for ActivityBurstinessExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.activity_burstiness import ActivityBurstinessExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_activity_burstiness_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = ActivityBurstinessExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
