# Comprehensive Unit Test for BollingerBandsUsageTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.bollinger_bands import BollingerBandsUsageTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_bollinger_bands_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = BollingerBandsUsageTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"bollinger_bands_transformed" in res.columns
    assert not res[f"bollinger_bands_transformed"].isnull().any()

def test_bollinger_bands_numerical_stability():
    transformer = BollingerBandsUsageTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"bollinger_bands_transformed"]).any()
