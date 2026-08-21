# Comprehensive Unit Test for PCADimensionalityReducer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.pca_dimensionality_reducer import PCADimensionalityReducer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_pca_dimensionality_reducer_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = PCADimensionalityReducer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"pca_dimensionality_reducer_transformed" in res.columns
    assert not res[f"pca_dimensionality_reducer_transformed"].isnull().any()

def test_pca_dimensionality_reducer_numerical_stability():
    transformer = PCADimensionalityReducer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"pca_dimensionality_reducer_transformed"]).any()
