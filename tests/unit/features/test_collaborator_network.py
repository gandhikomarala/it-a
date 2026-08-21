# Unit test for CollaboratorNetworkExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.collaborator_network import CollaboratorNetworkExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_collaborator_network_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = CollaboratorNetworkExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
