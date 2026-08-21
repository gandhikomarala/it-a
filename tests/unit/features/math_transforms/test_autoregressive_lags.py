# Comprehensive Unit Test for AutoregressiveLagTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.autoregressive_lags import AutoregressiveLagTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_autoregressive_lags_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = AutoregressiveLagTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"autoregressive_lags_transformed" in res.columns
    assert not res[f"autoregressive_lags_transformed"].isnull().any()

def test_autoregressive_lags_numerical_stability():
    transformer = AutoregressiveLagTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"autoregressive_lags_transformed"]).any()
