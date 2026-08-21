# Unit test for SupportEscalationExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.support_escalation import SupportEscalationExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_support_escalation_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = SupportEscalationExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
