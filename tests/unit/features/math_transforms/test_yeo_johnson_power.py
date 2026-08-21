# Comprehensive Unit Test for YeoJohnsonPowerTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.yeo_johnson_power import YeoJohnsonPowerTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_yeo_johnson_power_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = YeoJohnsonPowerTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"yeo_johnson_power_transformed" in res.columns
    assert not res[f"yeo_johnson_power_transformed"].isnull().any()

def test_yeo_johnson_power_numerical_stability():
    transformer = YeoJohnsonPowerTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"yeo_johnson_power_transformed"]).any()
