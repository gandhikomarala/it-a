# Unit test for DomainInteractionExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.transformations.interaction_terms import DomainInteractionExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_interaction_terms_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = DomainInteractionExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
