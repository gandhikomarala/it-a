# Comprehensive Unit Test for WaveletMultiscaleTransformer.
import pytest
import numpy as np
import pandas as pd
from ml.features.math_transforms.wavelet_multiscale import WaveletMultiscaleTransformer
from ml.data.synthetic_generator import SyntheticCustomerGenerator

def test_wavelet_multiscale_execution():
    gen = SyntheticCustomerGenerator(random_seed=42)
    df = gen.generate(45)
    
    transformer = WaveletMultiscaleTransformer()
    transformer.fit(df)
    res = transformer.transform(df)
    
    assert isinstance(res, pd.DataFrame)
    assert len(res) == 45
    assert f"wavelet_multiscale_transformed" in res.columns
    assert not res[f"wavelet_multiscale_transformed"].isnull().any()

def test_wavelet_multiscale_numerical_stability():
    transformer = WaveletMultiscaleTransformer()
    df_zeros = pd.DataFrame({"monthly_charge": np.zeros(20)})
    res = transformer.fit_transform(df_zeros)
    assert len(res) == 20
    assert not np.isinf(res[f"wavelet_multiscale_transformed"]).any()
