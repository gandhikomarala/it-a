# Unit test for NPSTrajectoryExtractor.
import pytest
import numpy as np
import pandas as pd
from ml.features.domain.nps_trajectory import NPSTrajectoryExtractor
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_nps_trajectory_fit_transform():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(50)
    
    extractor = NPSTrajectoryExtractor()
    transformed = extractor.fit_transform(df)
    
    assert isinstance(transformed, pd.DataFrame)
    assert len(transformed) == 50
    assert not transformed.empty
