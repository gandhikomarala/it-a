# Unit test for PaymentCadenceExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.payment_cadence import PaymentCadenceExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_payment_cadence_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = PaymentCadenceExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
