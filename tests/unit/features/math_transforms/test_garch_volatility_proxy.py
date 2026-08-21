# Comprehensive Unit Test for GARCHVolatilityProxyTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.garch_volatility_proxy import GARCHVolatilityProxyTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_garch_volatility_proxy_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = GARCHVolatilityProxyTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"garch_volatility_proxy_transformed" in res.columns
    assert not res[f"garch_volatility_proxy_transformed"].isnull().any()

def test_garch_volatility_proxy_numerical_stability():
    transformer = GARCHVolatilityProxyTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"garch_volatility_proxy_transformed"]).any()
