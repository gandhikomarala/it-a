# Comprehensive Unit Test for BoxCoxPowerTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.box_cox_power import BoxCoxPowerTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_box_cox_power_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = BoxCoxPowerTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"box_cox_power_transformed" in res.columns
    assert not res[f"box_cox_power_transformed"].isnull().any()

def test_box_cox_power_numerical_stability():
    transformer = BoxCoxPowerTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"box_cox_power_transformed"]).any()
