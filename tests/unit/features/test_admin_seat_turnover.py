# Unit test for AdminSeatTurnoverExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.admin_seat_turnover import AdminSeatTurnoverExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_admin_seat_turnover_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = AdminSeatTurnoverExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
