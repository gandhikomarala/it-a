# Unit test for ContractCliffExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.contract_cliff import ContractCliffExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_contract_cliff_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = ContractCliffExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
