# Comprehensive Unit Test for ExponentialMovingAverageTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.exponential_moving_average import ExponentialMovingAverageTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_exponential_moving_average_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = ExponentialMovingAverageTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"exponential_moving_average_transformed" in res.columns
    assert not res[f"exponential_moving_average_transformed"].isnull().any()

def test_exponential_moving_average_numerical_stability():
    transformer = ExponentialMovingAverageTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"exponential_moving_average_transformed"]).any()
