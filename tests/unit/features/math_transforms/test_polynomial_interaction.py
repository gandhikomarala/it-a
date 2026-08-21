# Comprehensive Unit Test for PolynomialInteractionTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.polynomial_interaction import PolynomialInteractionTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_polynomial_interaction_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = PolynomialInteractionTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"polynomial_interaction_transformed" in res.columns
    assert not res[f"polynomial_interaction_transformed"].isnull().any()

def test_polynomial_interaction_numerical_stability():
    transformer = PolynomialInteractionTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"polynomial_interaction_transformed"]).any()
