# Unit test for LifecycleCohortExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.transformations.lifecycle_cohort import LifecycleCohortExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_lifecycle_cohort_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = LifecycleCohortExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
