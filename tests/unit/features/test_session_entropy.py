# Unit test for SessionEntropyExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.session_entropy import SessionEntropyExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_session_entropy_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = SessionEntropyExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
