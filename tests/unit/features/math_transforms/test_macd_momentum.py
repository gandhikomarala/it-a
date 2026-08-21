# Comprehensive Unit Test for MACDMomentumTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.macd_momentum import MACDMomentumTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_macd_momentum_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = MACDMomentumTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"macd_momentum_transformed" in res.columns
    assert not res[f"macd_momentum_transformed"].isnull().any()

def test_macd_momentum_numerical_stability():
    transformer = MACDMomentumTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"macd_momentum_transformed"]).any()
