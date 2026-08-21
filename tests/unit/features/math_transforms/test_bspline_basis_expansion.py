# Comprehensive Unit Test for BSplineBasisExpansionTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.bspline_basis_expansion import BSplineBasisExpansionTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_bspline_basis_expansion_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = BSplineBasisExpansionTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"bspline_basis_expansion_transformed" in res.columns
    assert not res[f"bspline_basis_expansion_transformed"].isnull().any()

def test_bspline_basis_expansion_numerical_stability():
    transformer = BSplineBasisExpansionTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"bspline_basis_expansion_transformed"]).any()
