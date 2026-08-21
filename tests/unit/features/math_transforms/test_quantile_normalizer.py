# Comprehensive Unit Test for QuantileNormalizerTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.quantile_normalizer import QuantileNormalizerTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_quantile_normalizer_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = QuantileNormalizerTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"quantile_normalizer_transformed" in res.columns
    assert not res[f"quantile_normalizer_transformed"].isnull().any()

def test_quantile_normalizer_numerical_stability():
    transformer = QuantileNormalizerTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"quantile_normalizer_transformed"]).any()
