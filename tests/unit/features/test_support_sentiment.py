# Unit test for SupportRiskExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.transformations.support_sentiment import SupportRiskExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_support_sentiment_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = SupportRiskExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
