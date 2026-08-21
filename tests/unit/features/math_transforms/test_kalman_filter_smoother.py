# Comprehensive Unit Test for KalmanFilterUsageSmoother.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.kalman_filter_smoother import KalmanFilterUsageSmoother
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_kalman_filter_smoother_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = KalmanFilterUsageSmoother()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"kalman_filter_smoother_transformed" in res.columns
    assert not res[f"kalman_filter_smoother_transformed"].isnull().any()

def test_kalman_filter_smoother_numerical_stability():
    transformer = KalmanFilterUsageSmoother()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"kalman_filter_smoother_transformed"]).any()
