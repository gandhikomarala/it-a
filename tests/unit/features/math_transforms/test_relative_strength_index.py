# Comprehensive Unit Test for RSIUsageMomentumTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.relative_strength_index import RSIUsageMomentumTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_relative_strength_index_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = RSIUsageMomentumTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"relative_strength_index_transformed" in res.columns
    assert not res[f"relative_strength_index_transformed"].isnull().any()

def test_relative_strength_index_numerical_stability():
    transformer = RSIUsageMomentumTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"relative_strength_index_transformed"]).any()
